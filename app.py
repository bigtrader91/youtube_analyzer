"""
YouTube 콘텐츠 분석 시스템 - Streamlit 웹 인터페이스
"""
import os
import time
import pandas as pd
import streamlit as st
from pathlib import Path

# 백엔드 모듈 임포트
from src.api_clients import initialize_all_clients
from src.youtube_searcher import search_and_get_details
from src.comment_extractor import get_video_comments, comments_to_dataframe
from src.transcript_extractor import extract_and_clean_transcripts
from src.nlp_processor import KeywordExtractor
from src.data_manager import DataManager, ReportGenerator

# 페이지 기본 설정
st.set_page_config(
    page_title="YouTube 콘텐츠 분석 시스템",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'initialized_apis' not in st.session_state:
    st.session_state.initialized_apis = False
if 'youtube_search_results' not in st.session_state:
    st.session_state.youtube_search_results = None
if 'selected_video_ids' not in st.session_state:
    st.session_state.selected_video_ids = []
if 'transcripts_data' not in st.session_state:
    st.session_state.transcripts_data = {}
if 'comments_data' not in st.session_state:
    st.session_state.comments_data = {}
if 'nlp_results' not in st.session_state:
    st.session_state.nlp_results = {}

# 데이터 관리자 및 리포트 생성기 초기화
data_manager = DataManager()
report_generator = ReportGenerator(data_manager)

# 사이드바 구성
with st.sidebar:
    st.title("YouTube 분석 도구")
    
    # API 설정 (사이드바)
    st.header("API 설정")
    api_key_youtube = st.text_input("YouTube Data API 키", type="password")
    
    # 보안 경고
    st.warning("이 입력은 개발 및 테스트용입니다. 실제 배포 시에는 환경 변수나 Streamlit Secrets를 사용하세요.")
    
    # API 초기화 버튼
    if st.button("API 초기화"):
        # 환경 변수 설정 (임시)
        os.environ["YOUTUBE_API_KEY"] = api_key_youtube
        
        with st.spinner("API 클라이언트 초기화 중..."):
            youtube_success, _ = initialize_all_clients()
            
            if youtube_success:
                st.session_state.initialized_apis = True
                st.success("YouTube API 초기화 성공!")
            else:
                st.error("YouTube API 초기화 실패. API 키를 확인하세요.")
    
    # 기본 설정
    st.header("기본 설정")
    language_code = st.selectbox(
        "분석 언어",
        options=["ko", "en"],
        format_func=lambda x: "한국어" if x == "ko" else "영어",
        key="language_select"
    )
    
    # 앱 정보
    st.header("앱 정보")
    st.info(
        """
        이 앱은 YouTube Data API를 사용하여 동영상을 검색하고, 
        스크립트와 댓글을 추출하여 NLP 기법으로 키워드를 분석합니다.
        """
    )

# 메인 영역 구성
st.title("YouTube 콘텐츠 분석 시스템")

if not st.session_state.initialized_apis and not api_key_youtube:
    st.warning("사용을 시작하려면 사이드바에 YouTube API 키를 입력하고 초기화하세요.")

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs([
    "YouTube 검색", 
    "동영상 상세 분석", 
    "NLP 키워드 추출",
    "종합 리포트"
])

# 탭 1: YouTube 검색
with tab1:
    st.header("YouTube 검색")
    
    search_query = st.text_input("검색어 입력", placeholder="예: 파이썬 프로그래밍 강의")
    col1, col2 = st.columns(2)
    
    with col1:
        max_search_results = st.slider("최대 검색 결과 수", 5, 50, 10)
    
    with col2:
        content_type = st.radio("검색 유형", ["video", "channel"], format_func=lambda x: "동영상" if x == "video" else "채널")
    
    if st.button("YouTube 검색 실행"):
        if not st.session_state.initialized_apis:
            st.error("API가 초기화되지 않았습니다. 사이드바에서 API 키를 설정해주세요.")
        elif not search_query:
            st.error("검색어를 입력해주세요.")
        else:
            with st.spinner("YouTube 검색 중..."):
                try:
                    results_df = search_and_get_details(
                        query=search_query,
                        max_results=max_search_results,
                        content_type=content_type,
                        include_details=True
                    )
                    
                    # 결과 저장
                    if not results_df.empty:
                        data_manager.save_df_to_csv(
                            results_df,
                            f"search_results_{search_query.replace(' ', '_')}",
                            subdir="videos" if content_type == "video" else "channels"
                        )
                        st.session_state.youtube_search_results = results_df
                        st.success(f"YouTube 검색 완료! {len(results_df)}개 결과 찾음.")
                    else:
                        st.warning("검색 결과가 없습니다.")
                except Exception as e:
                    st.error(f"검색 중 오류 발생: {str(e)}")
    
    # 검색 결과 표시
    if st.session_state.youtube_search_results is not None:
        st.subheader("검색 결과")
        
        df_display = st.session_state.youtube_search_results
        
        if content_type == "video":
            # 동영상 검색 결과 표시
            display_cols = ['videoId', 'title', 'channelTitle', 'publishedAt', 'viewCount', 'likeCount', 'commentCount']
            display_cols = [col for col in display_cols if col in df_display.columns]
            
            # 결과 테이블 표시
            st.dataframe(df_display[display_cols], use_container_width=True)
            
            # 분석할 동영상 선택
            video_ids_to_analyze = st.multiselect(
                "분석할 동영상 ID 선택:",
                options=df_display['videoId'].dropna().unique().tolist(),
                default=[],
                key="video_select"
            )
            
            # 선택된 동영상 저장
            if st.button("선택한 동영상 분석 준비"):
                if video_ids_to_analyze:
                    st.session_state.selected_video_ids = video_ids_to_analyze
                    st.success(f"{len(video_ids_to_analyze)}개 동영상이 분석 준비되었습니다. '동영상 상세 분석' 탭으로 이동하세요.")
                else:
                    st.warning("분석할 동영상을 하나 이상 선택해주세요.")
        
        else:
            # 채널 검색 결과 표시
            display_cols = ['channelId', 'title', 'description', 'publishedAt', 'subscriberCount', 'videoCount', 'viewCount']
            display_cols = [col for col in display_cols if col in df_display.columns]
            
            st.dataframe(df_display[display_cols], use_container_width=True)

# 탭 2: 동영상 상세 분석
with tab2:
    st.header("동영상 상세 분석 (스크립트/댓글)")
    
    if not st.session_state.selected_video_ids:
        st.warning("YouTube 검색 탭에서 먼저 분석할 동영상을 선택해주세요.")
    else:
        st.write("선택된 동영상 ID:")
        for video_id in st.session_state.selected_video_ids:
            st.code(video_id)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fetch_comments = st.checkbox("댓글 수집", value=True)
            max_comments = st.slider("최대 댓글 수", 10, 500, 100)
        
        with col2:
            fetch_transcript = st.checkbox("스크립트 수집", value=True)
            preferred_languages = st.multiselect(
                "선호 언어 (스크립트)",
                options=["ko", "en", "ja", "zh-Hans", "zh-Hant", "es", "fr", "de"],
                default=["ko", "en"],
                format_func=lambda x: {
                    "ko": "한국어", "en": "영어", "ja": "일본어", 
                    "zh-Hans": "중국어(간체)", "zh-Hant": "중국어(번체)",
                    "es": "스페인어", "fr": "프랑스어", "de": "독일어"
                }.get(x, x)
            )
        
        if st.button("스크립트/댓글 추출 실행"):
            if not st.session_state.initialized_apis:
                st.error("API가 초기화되지 않았습니다. 사이드바에서 API 키를 설정해주세요.")
            else:
                all_transcripts = {}
                all_comments = {}
                
                for i, video_id in enumerate(st.session_state.selected_video_ids):
                    st.write(f"---\n**[{i+1}/{len(st.session_state.selected_video_ids)}] 동영상 '{video_id}' 처리 중...**")
                    
                    # 스크립트 추출
                    if fetch_transcript:
                        with st.spinner(f"스크립트 추출 중... ({video_id})"):
                            try:
                                transcript_df = extract_and_clean_transcripts(
                                    video_ids=[video_id],
                                    preferred_languages=preferred_languages
                                )
                                
                                if not transcript_df.empty:
                                    data_manager.save_df_to_csv(
                                        transcript_df,
                                        f"{video_id}_transcript",
                                        subdir="transcripts"
                                    )
                                    
                                    # 스크립트 텍스트 추출
                                    transcript_text = ""
                                    if 'cleaned_transcript' in transcript_df.columns:
                                        transcript_text = transcript_df['cleaned_transcript'].iloc[0]
                                    elif 'raw_transcript' in transcript_df.columns:
                                        transcript_text = transcript_df['raw_transcript'].iloc[0]
                                    
                                    all_transcripts[video_id] = {
                                        'text': transcript_text,
                                        'status': transcript_df['status'].iloc[0],
                                        'char_count': len(transcript_text) if transcript_text else 0
                                    }
                                    
                                    if transcript_df['status'].iloc[0] == 'success':
                                        st.success(f"스크립트 추출 성공! (글자 수: {len(transcript_text)})")
                                    else:
                                        reason = transcript_df['error_reason'].iloc[0] if 'error_reason' in transcript_df.columns else '알 수 없음'
                                        st.warning(f"스크립트 추출 실패: {reason}")
                                else:
                                    st.warning(f"스크립트를 찾을 수 없습니다. ({video_id})")
                                    all_transcripts[video_id] = {
                                        'text': "",
                                        'status': 'failed',
                                        'char_count': 0
                                    }
                            except Exception as e:
                                st.error(f"스크립트 추출 중 오류 발생: {str(e)}")
                                all_transcripts[video_id] = {
                                    'text': "",
                                    'status': 'error',
                                    'error': str(e),
                                    'char_count': 0
                                }
                    
                    # 댓글 수집
                    if fetch_comments:
                        with st.spinner(f"댓글 수집 중... ({video_id})"):
                            try:
                                comments = get_video_comments(
                                    video_id=video_id,
                                    max_total_comments=max_comments,
                                    fetch_replies=True
                                )
                                
                                if comments:
                                    comments_df = comments_to_dataframe(comments, include_replies=True)
                                    data_manager.save_df_to_csv(
                                        comments_df,
                                        f"{video_id}_comments",
                                        subdir="comments"
                                    )
                                    
                                    all_comments[video_id] = {
                                        'df': comments_df,
                                        'count': len(comments_df),
                                        'all_text': " ".join(comments_df['text'].dropna().astype(str).tolist())
                                    }
                                    
                                    st.success(f"댓글 수집 성공! (총 {len(comments_df)}개)")
                                else:
                                    st.warning(f"댓글이 없거나 수집 실패. ({video_id})")
                                    all_comments[video_id] = {
                                        'df': pd.DataFrame(),
                                        'count': 0,
                                        'all_text': ""
                                    }
                            except Exception as e:
                                st.error(f"댓글 수집 중 오류 발생: {str(e)}")
                                all_comments[video_id] = {
                                    'df': pd.DataFrame(),
                                    'count': 0,
                                    'all_text': "",
                                    'error': str(e)
                                }
                
                # 세션 상태에 저장
                st.session_state.transcripts_data = all_transcripts
                st.session_state.comments_data = all_comments
                
                st.success(f"모든 데이터 수집 완료! ({len(st.session_state.selected_video_ids)}개 동영상)")
        
        # 수집된 데이터 표시
        if st.session_state.transcripts_data or st.session_state.comments_data:
            st.subheader("수집된 데이터 미리보기")
            
            # 스크립트 미리보기
            if st.session_state.transcripts_data:
                st.write("**스크립트:**")
                for video_id, transcript in st.session_state.transcripts_data.items():
                    if transcript['status'] == 'success' and transcript['text']:
                        with st.expander(f"동영상 ID: {video_id} (글자 수: {transcript['char_count']})"):
                            st.text_area(
                                "스크립트", 
                                transcript['text'][:1000] + ("..." if len(transcript['text']) > 1000 else ""),
                                height=150
                            )
            
            # 댓글 미리보기
            if st.session_state.comments_data:
                st.write("**댓글 (상위 5개):**")
                for video_id, comment_data in st.session_state.comments_data.items():
                    if comment_data['count'] > 0:
                        with st.expander(f"동영상 ID: {video_id} (댓글 수: {comment_data['count']})"):
                            st.dataframe(comment_data['df'][['text', 'author', 'likeCount']].head())

# 탭 3: NLP 키워드 추출
with tab3:
    st.header("NLP 키워드 추출")
    
    if not st.session_state.selected_video_ids:
        st.warning("YouTube 검색 탭에서 먼저 분석할 동영상을 선택해주세요.")
    elif not st.session_state.transcripts_data and not st.session_state.comments_data:
        st.warning("'동영상 상세 분석' 탭에서 먼저 스크립트와 댓글을 수집해주세요.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            text_source_option = st.radio(
                "분석할 텍스트 소스 선택:", 
                ["스크립트", "댓글", "스크립트+댓글"],
                key="text_source"
            )
        
        with col2:
            nlp_methods = st.multiselect(
                "사용할 NLP 방법 선택:", 
                ["YAKE", "RAKE", "KeyBERT"],
                default=["YAKE", "RAKE"],
                format_func=lambda x: x,
                key="nlp_methods"
            )
        
        video_for_analysis = st.selectbox(
            "분석할 동영상 선택:",
            st.session_state.selected_video_ids,
            key="video_for_analysis"
        )
        
        top_n = st.slider("추출할 키워드 수", 5, 50, 20)
        
        # 분석 버튼
        if st.button("NLP 키워드 추출 실행"):
            if not video_for_analysis:
                st.error("분석할 동영상을 선택해주세요.")
            elif not nlp_methods:
                st.error("하나 이상의 NLP 방법을 선택해주세요.")
            else:
                # 선택된 텍스트 소스 가져오기
                text_to_process = ""
                source_name = ""
                
                if text_source_option == "스크립트" and video_for_analysis in st.session_state.transcripts_data:
                    text_to_process = st.session_state.transcripts_data[video_for_analysis].get('text', "")
                    source_name = "스크립트"
                
                elif text_source_option == "댓글" and video_for_analysis in st.session_state.comments_data:
                    text_to_process = st.session_state.comments_data[video_for_analysis].get('all_text', "")
                    source_name = "댓글"
                
                elif text_source_option == "스크립트+댓글":
                    transcript_text = ""
                    comment_text = ""
                    
                    if video_for_analysis in st.session_state.transcripts_data:
                        transcript_text = st.session_state.transcripts_data[video_for_analysis].get('text', "")
                    
                    if video_for_analysis in st.session_state.comments_data:
                        comment_text = st.session_state.comments_data[video_for_analysis].get('all_text', "")
                    
                    text_to_process = transcript_text + " " + comment_text
                    source_name = "스크립트+댓글"
                
                if not text_to_process.strip():
                    st.error(f"분석할 텍스트가 없습니다. 선택한 소스({text_source_option})에 내용이 비어 있습니다.")
                else:
                    with st.spinner("NLP 키워드 추출 중..."):
                        try:
                            # KeywordExtractor 초기화
                            language = st.session_state.get("language_select", "ko")
                            keyword_extractor = KeywordExtractor(language=language)
                            
                            # 방법명 변환 (대문자 -> 소문자)
                            methods_to_use = [method.lower() for method in nlp_methods]
                            
                            # 키워드 추출
                            nlp_results = keyword_extractor.extract_keywords(
                                text_to_process,
                                methods=methods_to_use,
                                top_n=top_n,
                                preprocess=True
                            )
                            
                            # 통합 키워드
                            combined_keywords = keyword_extractor.combine_keywords(
                                nlp_results,
                                top_n=top_n
                            )
                            
                            # 결과 저장
                            st.session_state.nlp_results[video_for_analysis] = {
                                'source': source_name,
                                'methods': methods_to_use,
                                'results': nlp_results,
                                'combined': combined_keywords
                            }
                            
                            # 분석 결과 저장
                            result_data = {
                                'video_id': video_for_analysis,
                                'text_source': source_name,
                                'text_length': len(text_to_process),
                                'keywords': nlp_results,
                                'combined_keywords': combined_keywords
                            }
                            
                            data_manager.save_dict_to_json(
                                result_data,
                                f"{video_for_analysis}_{source_name}_keywords",
                                subdir="analysis"
                            )
                            
                            st.success("NLP 키워드 추출 완료!")
                        except Exception as e:
                            st.error(f"키워드 추출 중 오류 발생: {str(e)}")
        
        # 결과 표시
        if video_for_analysis in st.session_state.nlp_results:
            result = st.session_state.nlp_results[video_for_analysis]
            st.subheader(f"키워드 추출 결과 - {result['source']}")
            
            # 통합 키워드 표시
            st.write("**통합 키워드 (상위 순위순)**")
            
            # 데이터프레임으로 변환
            df_combined = pd.DataFrame(result['combined'])
            
            # 데이터프레임에 필요한 컬럼이 있는지 확인하고 없으면 추가
            # 데이터 구조 디버깅
            if df_combined.empty:
                st.warning("추출된 키워드가 없습니다.")
            else:
                # 디버깅 정보
                st.write("디버깅 정보: 현재 데이터프레임 컬럼:", df_combined.columns.tolist())
                
                # 필요한 컬럼 확인 및 처리
                required_columns = ['keyword', 'score', 'methods', 'count']
                missing_columns = [col for col in required_columns if col not in df_combined.columns]
                
                if missing_columns:
                    st.warning(f"누락된 컬럼이 있습니다: {missing_columns}")
                    # 누락된 컬럼 추가
                    for col in missing_columns:
                        df_combined[col] = "N/A"
                
                # 존재하는 컬럼만 선택하여 표시
                existing_columns = [col for col in required_columns if col in df_combined.columns]
                if existing_columns:
                    df_combined = df_combined[existing_columns]
                    
                    # 컬럼명 변경
                    column_mapping = {
                        'keyword': '키워드',
                        'score': '점수',
                        'methods': '추출 방법',
                        'count': '빈도'
                    }
                    renamed_columns = {col: column_mapping.get(col, col) for col in existing_columns}
                    df_combined.rename(columns=renamed_columns, inplace=True)
                else:
                    # 모든 컬럼 표시
                    pass
                    
                # 테이블 표시
                st.dataframe(df_combined, use_container_width=True)
            
            # 개별 방법별 결과 표시
            st.write("**개별 방법별 결과**")
            
            tabs_methods = st.tabs([method.upper() for method in result['methods']])
            
            for i, method in enumerate(result['methods']):
                if method in result['results']:
                    with tabs_methods[i]:
                        keywords = result['results'][method]
                        df_method = pd.DataFrame(keywords, columns=['키워드', '점수'])
                        st.dataframe(df_method, use_container_width=True)
            
            # 워드클라우드 생성
            if st.button("워드클라우드 생성"):
                with st.spinner("워드클라우드 생성 중..."):
                    try:
                        wc_path = report_generator.generate_wordcloud(
                            result['combined'],
                            f"{video_for_analysis} - {result['source']} 키워드",
                            f"{video_for_analysis}_{result['source'].replace('+', '_')}_wordcloud",
                            subdir="reports"
                        )
                        
                        if wc_path:
                            st.success("워드클라우드 생성 완료!")
                            st.image(wc_path)
                        else:
                            st.error("워드클라우드 생성 실패.")
                    except Exception as e:
                        st.error(f"워드클라우드 생성 중 오류 발생: {str(e)}")

# 탭 4: 종합 리포트
with tab4:
    st.header("종합 리포트 생성")
    
    if not st.session_state.selected_video_ids:
        st.warning("YouTube 검색 탭에서 먼저 분석할 동영상을 선택해주세요.")
    else:
        # 리포트 생성할 동영상 선택
        videos_for_report = st.multiselect(
            "리포트 생성할 동영상 선택:",
            st.session_state.selected_video_ids,
            default=st.session_state.selected_video_ids,
            key="videos_for_report"
        )
        
        include_wordcloud = st.checkbox("워드클라우드 포함", value=True)
        
        if st.button("종합 리포트 생성"):
            if not videos_for_report:
                st.error("리포트를 생성할 동영상을 선택해주세요.")
            else:
                with st.spinner(f"총 {len(videos_for_report)}개 동영상 리포트 생성 중..."):
                    try:
                        # 일괄 분석 수행
                        batch_results = report_generator.generate_batch_analysis(
                            video_ids=videos_for_report,
                            include_wordclouds=include_wordcloud
                        )
                        
                        st.success(f"종합 리포트 생성 완료! (성공: {batch_results['successful_analyses']}, 실패: {batch_results['failed_analyses']})")
                        
                        # 리포트 결과 표시
                        for i, report_item in enumerate(batch_results['video_reports']):
                            video_id = report_item['video_id']
                            status = report_item['status']
                            
                            if status == 'success':
                                st.write(f"**{i+1}. {video_id}**: 성공 ✅")
                                
                                # 리포트 파일 로드
                                report_data = data_manager.load_dict_from_json(
                                    f"{video_id}_comprehensive_report",
                                    subdir="reports"
                                )
                                
                                if report_data:
                                    with st.expander(f"{video_id} 리포트 상세보기"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**스크립트 키워드 (상위 10개)**")
                                            if 'transcript_keywords' in report_data and report_data['transcript_keywords']:
                                                for j, kw in enumerate(report_data['transcript_keywords'][:10], 1):
                                                    st.write(f"{j}. {kw['keyword']} (점수: {kw['score']:.2f})")
                                            else:
                                                st.write("스크립트 키워드 없음")
                                        
                                        with col2:
                                            st.write("**댓글 키워드 (상위 10개)**")
                                            if 'comment_keywords' in report_data and report_data['comment_keywords']:
                                                for j, kw in enumerate(report_data['comment_keywords'][:10], 1):
                                                    st.write(f"{j}. {kw['keyword']} (점수: {kw['score']:.2f})")
                                            else:
                                                st.write("댓글 키워드 없음")
                                        
                                        # 워드클라우드 이미지 표시
                                        if include_wordcloud:
                                            if 'script_wordcloud_path' in report_data:
                                                st.write("**스크립트 워드클라우드**")
                                                st.image(report_data['script_wordcloud_path'])
                                            
                                            if 'comments_wordcloud_path' in report_data:
                                                st.write("**댓글 워드클라우드**")
                                                st.image(report_data['comments_wordcloud_path'])
                            else:
                                reason = report_item.get('reason', '알 수 없음')
                                st.write(f"**{i+1}. {video_id}**: 실패 ❌ - {reason}")
                    except Exception as e:
                        st.error(f"리포트 생성 중 오류 발생: {str(e)}")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>YouTube 콘텐츠 분석 시스템 - Streamlit 웹 인터페이스</p>
        <p>개발: YouTube Analyzer 프로젝트</p>
    </div>
    """,
    unsafe_allow_html=True
) 