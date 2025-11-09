import os
import numpy as np
import pandas as pd
import streamlit as st
import sqlalchemy
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(page_title='AI Talent Match — Prototype', layout='wide')

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_API_KEY:
  try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
  except Exception as e:
    st.sidebar.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
    llm = None
else:
  llm = None
  st.sidebar.markdown('*Note:* **GEMINI API key not found** in .env.')

try:
  engine = sqlalchemy.create_engine(DATABASE_URL, pool_pre_ping=True)
  with engine.connect():
    pass
except Exception as e:
  st.error(f"Failed to connect to database. Please check credentials and network connection. Error: {e}")
  st.stop()

st.sidebar.header('Vacancy inputs')
role_name = st.sidebar.text_input('Role name', 'Data Analyst')
job_level = st.sidebar.selectbox('Job level', ['Entry-Level (I)', 'Associate (II)', 'Specialist (III)', 'Senior (IV)', 'Principal (V)'], index=4)
role_purpose = st.sidebar.text_area('Role purpose', 'Turn raw data into actionable business insights to drive decision-making')
bench_ids_input = st.sidebar.text_input('Selected benchmark employee IDs (comma-separated)', 'DUP3015, EMP100044')
commit_benchmark = st.sidebar.checkbox('Save vacancy into talent_benchmarks table', value=False)
run_button = st.sidebar.button('Run matching')
def parse_bench_ids(text_ids):
  ids = [x.strip() for x in text_ids.split(',') if x.strip()]
  return ids

PARAM_SQL = '''
  WITH role_tb AS (
    SELECT * FROM "talent_benchmarks" WHERE job_vacancy_id = :job_vacancy_id
  ), 
  selected_benchmarks AS (
    SELECT employee_id
    FROM "master_dataset" 
    -- A bit different from the Step 2 answer (which was strictly 5) since there’s an input for the selected employee as a benchmark.
    WHERE employee_id = ANY(string_to_array(:selected_ids, ',')) 
  ),

  -- Algorithm
  benchmark_scores AS (
    SELECT
      m.tgv_name,
      es.tv_name,
      percentile_cont(0.5) WITHIN GROUP (ORDER BY es.score) AS baseline_score
    FROM "employee_score" es
    JOIN selected_benchmarks sb ON sb.employee_id = es.employee_id
    JOIN "tv_meta" m ON es.tv_name = m.tv_name 
    GROUP BY m.tgv_name, es.tv_name
  ),

  tv_match AS (
    SELECT
      e.employee_id,
      e.fullname, 
      e.directorate_name,
      e.position_name,
      e.grade_name,
      m.tgv_name,
      es.tv_name,
      m.feature_weight,
      b.baseline_score,
      es.score AS user_score,
      CASE
        WHEN b.baseline_score IS NULL THEN 0
        -- Boolean OHE categoricals
        WHEN es.tv_name LIKE 'disc_%' THEN
          CASE WHEN es.score = b.baseline_score THEN 100 ELSE 0 END
        -- Variables with inverse scale (lower is better), or rank-based TVs
        WHEN es.tv_name LIKE 'papi_z' OR es.tv_name LIKE 'papi_k' or es.tv_name LIKE 'strength_%' THEN
          ROUND(((2 * b.baseline_score - es.score) / NULLIF(b.baseline_score, 0) * 100)::numeric, 2)
        -- Numeric TVs
        WHEN es.tv_name NOT LIKE 'papi_z%' AND es.tv_name NOT LIKE 'papi_k%' AND es.tv_name NOT LIKE 'strength_%' AND es.tv_name NOT LIKE 'disc_%' THEN
          ROUND((es.score / NULLIF(b.baseline_score, 0) * 100)::numeric, 2)
        ELSE NULL
      END AS tv_match_rate,
      COALESCE(m.feature_weight, 1.0) AS tv_weight
    FROM "employee_score" es 
    JOIN "master_dataset" e ON e.employee_id = es.employee_id 
    JOIN "tv_meta" m ON es.tv_name = m.tv_name 
    LEFT JOIN benchmark_scores b ON b.tv_name = es.tv_name
  ),

  tgv_match AS (
    SELECT
      tm.employee_id,
      tm.tgv_name,
      ROUND(SUM(tm.tv_match_rate * tm.tv_weight) / NULLIF(SUM(tm.tv_weight),0), 2) AS tgv_match_rate,
      COALESCE((tb.weights_config ->> tm.tgv_name)::numeric, 1.0) AS tgv_weight_for_final
    FROM tv_match tm
    CROSS JOIN role_tb tb
    GROUP BY tm.employee_id, tm.tgv_name, tb.weights_config
  ),

  final_match AS (
    SELECT
      employee_id,
      ROUND(SUM(tgv_match_rate * tgv_weight_for_final) / NULLIF(SUM(tgv_weight_for_final),0), 2) AS final_match_rate
    FROM tgv_match
    GROUP BY employee_id
  )

  -- Final Output
  SELECT
    tm.employee_id,
    tm.fullname, 
    tm.directorate_name,
    tm.position_name AS role_name, 
    tm.grade_name,
    tm.tgv_name,
    tm.tv_name,
    CAST(tm.baseline_score AS TEXT) AS baseline_score, 
    CAST(tm.user_score AS TEXT) AS user_score,         
    tm.tv_match_rate,
    tg.tgv_match_rate,
    fm.final_match_rate
  FROM tv_match tm
  JOIN tgv_match tg
    ON tm.employee_id = tg.employee_id AND tm.tgv_name = tg.tgv_name
  JOIN final_match fm
    ON tm.employee_id = fm.employee_id
  ORDER BY fm.final_match_rate DESC, tm.employee_id, tm.tgv_name, tm.tv_name;
'''

def run_matching(selected_ids, role_name, job_level, role_purpose, commit_benchmark):
  selected_ids_literal = ','.join(selected_ids)
  job_vacancy_id = 1

  with engine.connect() as conn:
    transaction = conn.begin()
    try:
      if commit_benchmark:
        insert_sql = sqlalchemy.text("""
          INSERT INTO "talent_benchmarks" 
          ("role_name", "job_level", "role_purpose", "weights_config")
          VALUES (:role, :level, :purpose, '{}'::jsonb)
          RETURNING job_vacancy_id;
        """)
        res = conn.execute(
          insert_sql,
          role=role_name,
          level=job_level,
          purpose=role_purpose,
        ).fetchone()
        job_vacancy_id = res[0]
        st.success(f"New Vacancy Profile Saved. ID: {job_vacancy_id}")

      df = pd.read_sql(
        sqlalchemy.text(PARAM_SQL), 
        conn,
        params={'job_vacancy_id': job_vacancy_id, 'selected_ids': selected_ids_literal}
      )
      transaction.commit()
      return df, job_vacancy_id
            
    except Exception as e:
      transaction.rollback()
      st.error(f'SQL Query Failed. Error: {e}. Returning an empty result set.')
      empty_df = pd.DataFrame(columns=['employee_id', 'fullname', 'directorate_name', 'role_name', 'grade_name', 'tgv_name', 'tv_name', 'baseline_score', 'user_score', 'tv_match_rate', 'tgv_match_rate', 'final_match_rate'])
      return empty_df, job_vacancy_id

def plot_tgv_radar(df, top_fullname):
  df = df.groupby('tgv_name')['tgv_match_rate'].mean().reset_index()

  fig = go.Figure()
  fig.add_trace(go.Scatterpolar(
    r=df['tgv_match_rate'],
    theta=df['tgv_name'],
    fill='toself',
    name=f'Match Rate (%)',
    hovertemplate = '%{r:.2f}%<extra></extra>',
    line_color='deepskyblue'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 100],
        tickvals=[0, 25, 50, 75, 100],
        gridcolor='lightgrey'
      )),
    showlegend=False,
    title=f'TGV Match Rate for {top_fullname}',
    height=400,
    margin=dict(l=50, r=50, t=50, b=50)
  )
  return fig

def plot_tv_comparison(df):
  df['baseline_score'] = pd.to_numeric(df['baseline_score'], errors='coerce')
  df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')
  plot_df = pd.melt(df, id_vars=['tv_name'], value_vars=['baseline_score', 'user_score'], var_name='Score Type', value_name='Score')
  plot_df['Score Type'] = plot_df['Score Type'].replace({'baseline_score': 'Benchmark Target', 'user_score': 'Candidate Score'})
  
  fig = px.bar(
    plot_df, 
    x='tv_name', 
    y='Score', 
    color='Score Type', 
    barmode='group',
    text_auto='.2s',
    title='Score Comparison (Benchmark Target vs. Candidate Score)',
    color_discrete_map={'Benchmark Target': '#F63366', 'Candidate Score': '#00BFFF'} 
  )
  
  fig.update_layout(
    yaxis_title='Talent Value Score (1-5 Scale)',
    xaxis_title='Talent Value (TV)',
    legend_title='Score Type',
    yaxis=dict(range=[0, 5.5], tickvals=[1, 2, 3, 4, 5]),
    height=400
  )
  return fig

if run_button:
  st.title(f'AI Talent Match — {role_name} ({job_level})')
  st.subheader('Dashboard powered by Success Formula and AI')
  st.markdown('---')
  bench_ids = parse_bench_ids(bench_ids_input)
  if len(bench_ids) == 0:
    st.error('Please provide at least one benchmark employee ID.')
  elif not role_name.strip():
    st.error('Please provide a role name.')
  elif not role_purpose.strip():
    st.error('Please provide a role description.')
  else:
    with st.spinner('Running application logic...'):
      try:
        df, returned_job_id = run_matching(
          bench_ids, 
          role_name=role_name, 
          job_level=job_level, 
          role_purpose=role_purpose, 
          commit_benchmark=commit_benchmark
        )
      except Exception as e:
        st.error(f'Application failed during core logic execution: {e}')
        st.stop()
      
      if df.empty:
        st.error("No data returned from the query. Check your database connection, the SQL query, or the benchmark IDs.")
        st.stop()

      ranked_all = df[['employee_id','fullname','directorate_name','role_name','grade_name','final_match_rate']].drop_duplicates()
      ranked_all['final_match_rate'] = pd.to_numeric(ranked_all['final_match_rate'], errors='coerce')
      ranked = ranked_all.sort_values('final_match_rate', ascending=False)
      
      # --- ROW 1: AI PROFILE & KEY METRICS ---
      st.header('AI Job Profile & Key Metrics')
      col1, col2 = st.columns([7, 3])
      with col1:
        # AI-Generated Job Profile
        if llm:
          try:
            tgv_prompt_list = ", ".join(df['tgv_name'].unique().tolist() if not df.empty else ['High Performance', 'Drive', 'Analytical Thinking'])
            messages = [SystemMessage(
                content="You are a concise, enterprise-grade job profile generator. Stick strictly to the requested format."
              ),
              HumanMessage(
                content=f"""
                  Generate a job profile for the role '{role_name}' (level {job_level}) with purpose: "{role_purpose}".
                  The role's Success Pattern is weighted heavily towards the following Talent Group Variables (TGVs): {tgv_prompt_list}.

                  Output must include:
                  1. Role Summary (1-2 sentences).
                  2. Core Mission (1 sentence, based on purpose).
                  3. Key Competencies Required (6 bullet points, aligned to the provided TGVs).
                """
              )
            ]
            response = llm.invoke(messages)
            st.markdown(response.content)
          except Exception as e:
            st.info(f'Gemini API request failed via LangChain: {e}. Please ensure your API key is correct and valid.')
        else:
          st.info("GEMINI API key not set or model initialization failed. Cannot generate AI profile.")
      with col2:
        total_employees = len(ranked)
        avg_match_rate = ranked['final_match_rate'].mean() if not ranked.empty else 0
        top_match = ranked.iloc[0]['final_match_rate'] if not ranked.empty else 0
        st.metric(label="Total Talent Pool Evaluated", value=total_employees)
        st.metric(label="Average Match Rate (Pool)", value=f"{avg_match_rate:.2f}%")
        st.metric(label="Highest Match Rate Found", value=f"{top_match:.2f}%")
      st.markdown('---')

      # --- ROW 2: RANKED LIST AND DISTRIBUTION ---
      st.header('Talent Pool Analysis')
      col3, col4 = st.columns([4, 6])
      with col3:
        st.subheader('Ranked Candidates (Top 200)')
        table_cols = ['fullname','final_match_rate', 'role_name', 'directorate_name', 'grade_name']
        st.dataframe(
          ranked[table_cols].rename(columns={'fullname': 'NAME', 'final_match_rate': 'MATCH RATE', 'role_name': 'Role', 'directorate_name': 'Directorate', 'grade_name': 'Job_Level'}).head(200).style.format({'MATCH RATE': '{:.2f}%'}),
          use_container_width=True,
          hide_index=True
        )
      with col4:
        st.subheader('Final Match Rate Distribution')
        fig = px.histogram(
          ranked_all, 
          x='final_match_rate', 
          nbins=25, 
          title='Distribution of Final Match Scores Across Talent Pool',
          color_discrete_sequence=['#4c78a8'] 
        )
        fig.update_layout(template="plotly_white", xaxis_title="Final Match Rate (%)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
      st.markdown('---')

      # --- ROW 3: TOP CANDIDATE DETAIL AND COMPARISON VISUALS ---
      st.header('Top Candidate Deep Dive')
      top_emp = ranked.iloc[0]['employee_id'] if not ranked.empty else None
      if top_emp:
        detail = df[df['employee_id']==top_emp].sort_values(['tgv_name','tv_name'])
        top_fullname = detail.iloc[0]['fullname']
        col5, col6 = st.columns([4, 6])
        with col5:
          st.subheader(f'Detail: {top_fullname}')
          st.caption(f'Final Match Rate: *{detail.iloc[0]["final_match_rate"]:.2f}%*')
          detail_table = detail[['tgv_name','tv_name','baseline_score','user_score','tv_match_rate']].rename(
            columns={'tgv_name': 'TGV', 'tv_name': 'TV', 'baseline_score': 'Target Score', 'user_score': 'Cand. Score', 'tv_match_rate': 'TV Match Rate'}
          )
          st.dataframe(
            detail_table.style.format({'TV Match Rate': '{:.2f}%'}),
            use_container_width=True,
            hide_index=True
          )
          st.plotly_chart(plot_tgv_radar(detail, top_fullname), use_container_width=True)
          st.markdown('### Match Summary & Key Takeaways 📈')
          if llm:
            try:
              top_candidate_scores = detail[['tgv_name', 'tgv_match_rate']].drop_duplicates().to_string(index=False, header=True)
              summary_messages = [
                SystemMessage(
                  content="You are a concise HR/Talent Analyst. Your goal is to provide a brief narrative summary and key takeaways for a candidate's talent match against a benchmark. Use professional language."
                ),
                HumanMessage(
                  content=f"""
                  Analyze the following TGV (Talent Group Variable) match rates for candidate **{top_fullname}** against the **{role_name}** benchmark:
                  {top_candidate_scores}
                  Based on these scores, generate a summary (2-3 sentences) followed by three key bullet points outlining the candidate's strongest match areas and potential gaps/development needs. Focus on the TGVs with the highest and lowest match rates.
                  """
                )
              ]
              summary_response = llm.invoke(summary_messages)
              st.markdown(summary_response.content)
            except Exception as e:
              st.info(f"AI Summary generation failed: {e}. Check API key and service status.")
          else:
            st.info("GEMINI API key not set or model initialization failed. Cannot generate AI summary.")
        with col6:
          st.subheader('Score Comparison')
          st.plotly_chart(plot_tv_comparison(detail), use_container_width=True)
      else:
        st.info("No candidates were found in the data to analyze.")