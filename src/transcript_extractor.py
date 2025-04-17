"""
YouTube 동영상 스크립트 추출 모듈 - youtube-transcript-api 활용
"""
import time
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from src.api_clients import youtube_api


def get_video_transcripts(
    video_ids: List[str],
    preferred_languages: List[str] = ['ko', 'en']
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    주어진 YouTube 동영상 ID 목록에 대해 스크립트를 추출합니다.
    
    Args:
        video_ids (List[str]): 스크립트를 추출할 YouTube 동영상 ID 목록
        preferred_languages (List[str], optional): 선호하는 언어 코드 목록 (기본값: ['ko', 'en'])
            먼저 나열된 언어가 우선적으로 시도됩니다.
            
    Returns:
        Tuple[Dict[str, str], List[Dict[str, str]]]: 
            (성공적으로 추출된 스크립트 데이터 딕셔너리, 실패한 동영상 정보 목록)
            스크립트 데이터 딕셔너리: {video_id: transcript_text, ...}
            실패 목록: [{'videoId': video_id, 'reason': error_reason}, ...]
    """
    transcripts_data = {}
    failed_videos = []
    
    print(f"총 {len(video_ids)}개 비디오의 스크립트 추출 시작...")
    
    for video_id in video_ids:
        try:
            print(f"  비디오 ID '{video_id}' 스크립트 추출 시도...")
            # API 호출 사이에 약간의 지연 추가 (서버 부하 방지 목적)
            time.sleep(0.5)
            
            # 사용 가능한 자막 목록 가져오기
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            target_transcript = None
            
            # 1. 사용자가 생성한 수동 스크립트 중 선호 언어 확인
            manual_langs = {t.language_code for t in transcript_list if not t.is_generated}
            for lang in preferred_languages:
                if lang in manual_langs:
                    print(f"    '{lang}' 언어의 수동 스크립트 발견.")
                    target_transcript = transcript_list.find_manually_created_transcript([lang])
                    break
            
            # 2. 수동 스크립트 없으면, 자동 생성된 스크립트 중 선호 언어 확인
            if not target_transcript:
                generated_langs = {t.language_code for t in transcript_list if t.is_generated}
                for lang in preferred_languages:
                    if lang in generated_langs:
                        print(f"    '{lang}' 언어의 자동 생성 스크립트 발견.")
                        target_transcript = transcript_list.find_generated_transcript([lang])
                        break
            
            # 3. 선호 언어 스크립트 없으면, 사용 가능한 아무 스크립트나 가져오기 시도
            if not target_transcript:
                print(f"    선호 언어({preferred_languages}) 스크립트 없음. 사용 가능한 스크립트 가져오기 시도...")
                try:
                    # 첫 번째 선호 언어를 기준으로 찾기
                    target_transcript = transcript_list.find_transcript(preferred_languages)
                except NoTranscriptFound:
                    # 아무 언어나 가져오기 시도
                    print("    어떤 선호 언어로도 스크립트를 찾을 수 없어 자동 선택 시도...")
                    available_transcripts = list(transcript_list._transcripts.values())
                    if available_transcripts:
                        target_transcript = available_transcripts[0]
                        print(f"    '{target_transcript.language_code}' 언어의 스크립트 자동 선택됨.")
                    else:
                        raise NoTranscriptFound("사용 가능한 스크립트가 없습니다.")
            
            # 스크립트 추출 및 텍스트 병합
            transcript_segments = target_transcript.fetch()
            
            # 텍스트 전처리: 줄바꿈 제거, 여러 공백 단일화
            full_transcript = " ".join([segment['text'] for segment in transcript_segments])
            full_transcript = full_transcript.replace('\n', ' ').replace('\r', ' ')
            # 여러 개의 공백을 하나로 압축
            import re
            full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
            
            transcripts_data[video_id] = full_transcript
            print(f"    비디오 ID '{video_id}' 스크립트 추출 성공 (길이: {len(full_transcript)} 자).")
            
        except TranscriptsDisabled:
            print(f"    오류: 비디오 ID '{video_id}'의 스크립트(자막)가 비활성화되었습니다.")
            failed_videos.append({'videoId': video_id, 'reason': 'TranscriptsDisabled'})
        except NoTranscriptFound as e:
            print(f"    오류: 비디오 ID '{video_id}'에서 스크립트를 찾을 수 없습니다. ({e})")
            failed_videos.append({'videoId': video_id, 'reason': f'NoTranscriptFound: {e}'})
        except VideoUnavailable:
            print(f"    오류: 비디오 ID '{video_id}'는 더 이상 사용할 수 없거나 존재하지 않습니다.")
            failed_videos.append({'videoId': video_id, 'reason': 'VideoUnavailable'})
        except Exception as e:
            print(f"    오류: 비디오 ID '{video_id}' 처리 중 예외 발생: {e}")
            failed_videos.append({'videoId': video_id, 'reason': f'Exception: {e}'})
    
    print(f"\n총 {len(transcripts_data)}개의 스크립트 추출 완료.")
    if failed_videos:
        print(f"{len(failed_videos)}개의 비디오에서 스크립트 추출 실패.")
    
    return transcripts_data, failed_videos


def clean_transcript(transcript: str, remove_pattern: Optional[List[str]] = None) -> str:
    """
    스크립트 텍스트를 정제합니다.
    
    Args:
        transcript (str): 원본 스크립트 텍스트
        remove_pattern (List[str], optional): 제거할 패턴 목록 (예: ['[음악]', '[박수]'])
        
    Returns:
        str: 정제된 스크립트 텍스트
    """
    import re
    
    # 기본 제거 패턴
    default_patterns = [
        r'\[음악\]', r'\[박수\]', r'\[웃음\]', r'\[소음\]', r'\[잡음\]',
        r'\[배경 음악\]', r'\[음악 재생\]', r'\[광고\]', r'\[침묵\]'
    ]
    
    # 사용자 정의 패턴 추가
    if remove_pattern:
        # 문자열 그대로가 아니라 정규식 패턴으로 변환
        patterns = default_patterns + [rf'\{p}' if p.startswith('[') else rf'{p}' for p in remove_pattern]
    else:
        patterns = default_patterns
    
    # 모든 패턴 제거
    cleaned_text = transcript
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    # 줄바꿈 문자를 공백으로 변환
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
    
    # 여러 개의 공백을 하나로 압축
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def extract_and_clean_transcripts(
    video_ids: List[str],
    preferred_languages: List[str] = ['ko', 'en'],
    clean_patterns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    YouTube 동영상 ID 목록에서 스크립트를 추출하고 정제한 후 DataFrame으로 반환합니다.
    
    Args:
        video_ids (List[str]): 스크립트를 추출할 YouTube 동영상 ID 목록
        preferred_languages (List[str], optional): 선호하는 언어 코드 목록 (기본값: ['ko', 'en'])
        clean_patterns (List[str], optional): 제거할 추가 패턴 목록
        
    Returns:
        pd.DataFrame: 추출 및 정제된 스크립트 데이터
    """
    # 스크립트 추출
    transcripts_data, failed_videos = get_video_transcripts(
        video_ids=video_ids,
        preferred_languages=preferred_languages
    )
    
    # 결과 DataFrame 준비
    result_rows = []
    
    # 성공한 추출 데이터 처리
    for video_id, transcript in transcripts_data.items():
        # 스크립트 정제
        cleaned_transcript = clean_transcript(transcript, clean_patterns)
        
        result_rows.append({
            'videoId': video_id,
            'raw_transcript': transcript,
            'cleaned_transcript': cleaned_transcript,
            'char_count': len(cleaned_transcript),
            'word_count': len(cleaned_transcript.split()),
            'status': 'success'
        })
    
    # 실패한 추출 데이터 처리
    for failed in failed_videos:
        result_rows.append({
            'videoId': failed['videoId'],
            'raw_transcript': '',
            'cleaned_transcript': '',
            'char_count': 0,
            'word_count': 0,
            'status': 'failed',
            'error_reason': failed['reason']
        })
    
    # DataFrame 생성 및 반환
    return pd.DataFrame(result_rows)


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 동영상 ID 샘플 (실제 예제)
    TEST_VIDEO_IDS = [
        "ogfYd705cRs",  # "파이썬 기초 강의" 관련 동영상 (예시)
        "k5_38U27-Xc",  # "데이터 분석 강의" 관련 동영상 (예시)
        "dQw4w9WgXcQ",  # 유명한 뮤직비디오 (예시)
        "non_existent_video_id"  # 존재하지 않는 ID (실패 사례 테스트용)
    ]
    
    print("YouTube 동영상 스크립트 추출 테스트")
    
    # 추출 및 정제된 스크립트 데이터 가져오기
    result_df = extract_and_clean_transcripts(
        video_ids=TEST_VIDEO_IDS,
        preferred_languages=['ko', 'en', 'ja'],  # 한국어, 영어, 일본어 순으로 선호
        clean_patterns=['[광고]', '[음악 시작]']  # 추가 정제 패턴
    )
    
    # 결과 출력
    if not result_df.empty:
        # 성공한 추출만 필터링
        success_df = result_df[result_df['status'] == 'success']
        failed_df = result_df[result_df['status'] == 'failed']
        
        print("\n--- 추출 성공한 스크립트 ---")
        if not success_df.empty:
            for _, row in success_df.iterrows():
                print(f"비디오 ID: {row['videoId']}")
                print(f"단어 수: {row['word_count']}, 글자 수: {row['char_count']}")
                # 너무 길면 일부만 출력
                transcript_preview = row['cleaned_transcript'][:200] + "..." if len(row['cleaned_transcript']) > 200 else row['cleaned_transcript']
                print(f"스크립트 미리보기: {transcript_preview}")
                print("-" * 50)
        else:
            print("성공적으로 추출된 스크립트가 없습니다.")
        
        print("\n--- 추출 실패한 스크립트 ---")
        if not failed_df.empty:
            print(failed_df[['videoId', 'error_reason']])
        else:
            print("실패한 스크립트 추출이 없습니다.")
        
        # 결과를 파일로 저장 (주석 해제하여 사용 가능)
        # result_df.to_csv("video_transcripts_results.csv", index=False, encoding='utf-8-sig') 