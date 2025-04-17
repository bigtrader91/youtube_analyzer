"""
YouTube 콘텐츠 분석 시스템 - 메인 실행 스크립트
"""
import argparse
import sys
from typing import Dict, List, Any, Optional, Tuple, Union

from src.api_clients import initialize_all_clients
from src.youtube_searcher import search_and_get_details
from src.comment_extractor import get_video_comments, comments_to_dataframe
from src.transcript_extractor import extract_and_clean_transcripts
from src.nlp_processor import KeywordExtractor
from src.data_manager import DataManager, ReportGenerator


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='유튜브 콘텐츠 분석 시스템')
    
    # 기본 모드 지정
    parser.add_argument(
        '--mode',
        type=str,
        choices=['search', 'analyze', 'report', 'complete'],
        default='complete',
        help='실행 모드 (search: 검색만, analyze: 분석만, report: 리포트 생성만, complete: 전체 과정)'
    )
    
    # 검색 관련 인수
    parser.add_argument('--query', type=str, help='유튜브 검색 키워드')
    parser.add_argument('--max_results', type=int, default=10, help='검색 결과 최대 수')
    
    # 특정 비디오 분석용 인수
    parser.add_argument('--video_ids', type=str, nargs='+', help='분석할 비디오 ID 목록')
    
    # 댓글 관련 인수
    parser.add_argument('--max_comments', type=int, default=50, help='수집할 댓글 수')
    parser.add_argument('--skip_comments', action='store_true', help='댓글 수집 건너뛰기')
    
    # 스크립트 관련 인수
    parser.add_argument('--skip_transcript', action='store_true', help='스크립트 수집 건너뛰기')
    
    # 분석 관련 인수
    parser.add_argument('--language', type=str, default='ko', help='분석 언어 (기본: 한국어)')
    
    # 리포트 관련 인수
    parser.add_argument('--no_wordcloud', action='store_true', help='워드클라우드 생성 건너뛰기')
    
    return parser.parse_args()


def search_videos(args):
    """검색 모드: 키워드로 동영상 검색"""
    if not args.query:
        print("오류: 검색 모드에서는 --query 인수가 필요합니다.")
        return None
    
    print(f"'{args.query}' 키워드로 유튜브 검색 시작 (최대 {args.max_results}개 결과)...")
    
    # 검색 실행
    results_df = search_and_get_details(
        query=args.query,
        max_results=args.max_results,
        content_type="video",
        include_details=True
    )
    
    if results_df.empty:
        print("검색 결과가 없습니다.")
        return None
    
    # 결과 저장
    data_manager = DataManager()
    save_path = data_manager.save_df_to_csv(
        results_df,
        f"search_results_{args.query.replace(' ', '_')}",
        subdir="videos"
    )
    
    print(f"총 {len(results_df)}개 비디오 검색 결과 저장 완료.")
    
    # 검색된 비디오 ID 목록 반환
    if 'videoId' in results_df.columns:
        return results_df['videoId'].tolist()
    return None


def collect_video_data(video_ids, args):
    """비디오 데이터 수집: 댓글 및 스크립트"""
    if not video_ids:
        print("오류: 비디오 ID가 없습니다.")
        return
    
    data_manager = DataManager()
    
    for i, video_id in enumerate(video_ids):
        print(f"\n[{i+1}/{len(video_ids)}] 비디오 ID '{video_id}' 데이터 수집 중...")
        
        # 댓글 수집
        if not args.skip_comments:
            print(f"댓글 수집 중 (최대 {args.max_comments}개)...")
            comments = get_video_comments(
                video_id=video_id,
                max_total_comments=args.max_comments,
                fetch_replies=True
            )
            
            if comments:
                # DataFrame으로 변환 및 저장
                comments_df = comments_to_dataframe(comments, include_replies=True)
                data_manager.save_df_to_csv(
                    comments_df,
                    f"{video_id}_comments",
                    subdir="comments"
                )
                print(f"총 {len(comments_df)}개 댓글/답글 저장 완료.")
            else:
                print("댓글이 없거나 수집 실패.")
        
        # 스크립트 수집
        if not args.skip_transcript:
            print(f"스크립트 수집 중...")
            transcript_df = extract_and_clean_transcripts(
                video_ids=[video_id],
                preferred_languages=['ko', 'en']
            )
            
            if not transcript_df.empty:
                data_manager.save_df_to_csv(
                    transcript_df,
                    f"{video_id}_transcript",
                    subdir="transcripts"
                )
                
                status = transcript_df['status'].iloc[0]
                if status == 'success':
                    print(f"스크립트 저장 완료 (글자 수: {transcript_df['char_count'].iloc[0]}).")
                else:
                    reason = transcript_df['error_reason'].iloc[0] if 'error_reason' in transcript_df.columns else '알 수 없음'
                    print(f"스크립트 추출 실패: {reason}")
            else:
                print("스크립트 수집 실패.")


def analyze_video_data(video_ids, args):
    """비디오 데이터 분석: 댓글 및 스크립트 분석"""
    if not video_ids:
        print("오류: 비디오 ID가 없습니다.")
        return
    
    report_generator = ReportGenerator()
    
    for i, video_id in enumerate(video_ids):
        print(f"\n[{i+1}/{len(video_ids)}] 비디오 ID '{video_id}' 데이터 분석 중...")
        
        # 댓글 분석
        if not args.skip_comments:
            print("댓글 분석 중...")
            comment_analysis = report_generator.analyze_comments(
                video_id=video_id,
                language=args.language
            )
            
            if comment_analysis:
                print(f"댓글 분석 완료.")
                if 'combined_keywords' in comment_analysis:
                    for j, kw in enumerate(comment_analysis['combined_keywords'][:5], 1):
                        print(f"  {j}. {kw['keyword']} (점수: {kw['score']:.2f})")
            else:
                print("댓글 분석 실패 또는 분석할 데이터 없음.")
        
        # 스크립트 분석
        if not args.skip_transcript:
            print("스크립트 분석 중...")
            transcript_analysis = report_generator.analyze_transcript(
                video_id=video_id,
                language=args.language
            )
            
            if transcript_analysis:
                print(f"스크립트 분석 완료.")
                if 'combined_keywords' in transcript_analysis:
                    for j, kw in enumerate(transcript_analysis['combined_keywords'][:5], 1):
                        print(f"  {j}. {kw['keyword']} (점수: {kw['score']:.2f})")
            else:
                print("스크립트 분석 실패 또는 분석할 데이터 없음.")


def generate_reports(video_ids, args):
    """비디오 리포트 생성"""
    if not video_ids:
        print("오류: 비디오 ID가 없습니다.")
        return
    
    print(f"\n총 {len(video_ids)}개 비디오 리포트 생성 중...")
    
    report_generator = ReportGenerator()
    report_generator.generate_batch_analysis(
        video_ids=video_ids,
        include_wordclouds=not args.no_wordcloud
    )


def main():
    """메인 실행 함수"""
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # API 클라이언트 초기화
    print("API 클라이언트 초기화 중...")
    youtube_success, _ = initialize_all_clients()
    
    if not youtube_success:
        print("오류: YouTube API 클라이언트 초기화 실패")
        sys.exit(1)
    
    # 비디오 ID 목록 준비
    video_ids = None
    
    # 모드별 처리
    if args.mode in ['search', 'complete']:
        # 검색 수행
        video_ids = search_videos(args)
    elif args.mode in ['analyze', 'report']:
        # 비디오 ID 직접 지정
        if args.video_ids:
            video_ids = args.video_ids
        else:
            print("오류: analyze/report 모드에서는 --video_ids 인수가 필요합니다.")
            sys.exit(1)
    
    if not video_ids:
        print("처리할 비디오 ID가 없습니다.")
        sys.exit(1)
    
    print(f"처리할 비디오 ID: {video_ids}")
    
    # 데이터 수집 (search/complete 모드)
    if args.mode in ['complete']:
        collect_video_data(video_ids, args)
    
    # 데이터 분석 (analyze/complete 모드)
    if args.mode in ['analyze', 'complete']:
        analyze_video_data(video_ids, args)
    
    # 리포트 생성 (report/complete 모드)
    if args.mode in ['report', 'complete']:
        generate_reports(video_ids, args)
    
    print("\n작업 완료!")


if __name__ == "__main__":
    main() 