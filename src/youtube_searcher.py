"""
YouTube 영상/채널 검색 및 기본 정보 수집 모듈 - YouTube Data API 활용
"""
import time
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
from googleapiclient.errors import HttpError

from src.api_clients import youtube_api


def search_youtube(
    query: str,
    max_total_results: int = 50,
    max_results_per_page: int = 50,
    content_type: str = "video,channel",
    order: str = "relevance",
    region_code: str = "KR",
    relevance_language: str = "ko",
) -> List[Dict[str, Any]]:
    """
    YouTube Data API를 사용하여 동영상 및 채널을 검색하고 기본 정보를 반환합니다.
    
    Args:
        query (str): 검색할 키워드 문자열
        max_total_results (int, optional): 총 반환할 결과 수 (기본값: 50)
        max_results_per_page (int, optional): 페이지당 최대 결과 수 (기본값: 50, 최대 50)
        content_type (str, optional): 검색할 콘텐츠 유형 (기본값: "video,channel")
            가능한 값: "video", "channel", "playlist" 또는 조합 (쉼표로 구분)
        order (str, optional): 결과 정렬 방식 (기본값: "relevance")
            가능한 값: "relevance", "date", "rating", "title", "videoCount", "viewCount"
        region_code (str, optional): 지역 코드 (기본값: "KR" - 대한민국)
        relevance_language (str, optional): 관련성 언어 (기본값: "ko" - 한국어)
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    youtube_service = youtube_api.get_service()
    if not youtube_service:
        print("오류: YouTube API 서비스 초기화에 실패했습니다.")
        return []
    
    all_results = []
    next_page_token = None
    
    print(f"'{query}' 키워드로 유튜브 검색 시작...")
    
    while len(all_results) < max_total_results:
        try:
            current_max_results = min(max_results_per_page, max_total_results - len(all_results))
            if current_max_results <= 0:
                break
            
            request = youtube_service.search().list(
                part="snippet",
                q=query,
                type=content_type,
                order=order,
                maxResults=current_max_results,
                regionCode=region_code,
                relevanceLanguage=relevance_language,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get("items", []):
                result_item = {
                    'kind': item['id']['kind'],  # 'youtube#video' or 'youtube#channel'
                    'publishedAt': item['snippet']['publishedAt'],
                    'channelId': item['snippet']['channelId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'thumbnail_default': item['snippet']['thumbnails']['default']['url'],
                    'channelTitle': item['snippet']['channelTitle'],
                }
                
                # 동영상인 경우 videoId 추가
                if item['id']['kind'] == 'youtube#video':
                    result_item['videoId'] = item['id']['videoId']
                # 채널인 경우 id의 channelId 추가
                elif item['id']['kind'] == 'youtube#channel':
                    result_item['channelId_item'] = item['id']['channelId']
                
                all_results.append(result_item)
                if len(all_results) >= max_total_results:
                    break  # 목표 결과 수 도달 시 중단
            
            next_page_token = response.get("nextPageToken")
            print(f"현재까지 수집된 결과 수: {len(all_results)}")
            
            if not next_page_token:
                print("더 이상 결과 페이지가 없습니다.")
                break  # 다음 페이지 토큰 없으면 종료
            
            # API 할당량 고려하여 약간의 지연 추가
            time.sleep(0.5)
            
        except HttpError as e:
            print(f"API 요청 중 오류 발생: {e}")
            if e.resp.status == 403:
                print("오류 403: 할당량 초과 또는 API 키/권한 문제일 수 있습니다.")
            break  # 오류 발생 시 중단
        except Exception as e:
            print(f"알 수 없는 오류 발생: {e}")
            break
    
    print(f"총 {len(all_results)}개의 검색 결과를 수집했습니다.")
    return all_results


def get_video_details(
    video_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    videos.list API를 사용하여 비디오의 상세 정보를 가져옵니다.
    
    Args:
        video_ids (List[str]): 정보를 가져올 비디오 ID 목록
        
    Returns:
        List[Dict[str, Any]]: 비디오 상세 정보 목록
    """
    youtube_service = youtube_api.get_service()
    if not youtube_service:
        print("오류: YouTube API 서비스 초기화에 실패했습니다.")
        return []
    
    # 50개씩 나누어 처리 (API 제한)
    batch_size = 50
    all_video_details = []
    
    for i in range(0, len(video_ids), batch_size):
        batch_ids = video_ids[i:i + batch_size]
        print(f"비디오 상세 정보 조회 중: {i+1}~{min(i+batch_size, len(video_ids))} / {len(video_ids)}")
        
        try:
            request = youtube_service.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch_ids)
            )
            response = request.execute()
            
            for item in response.get("items", []):
                video_detail = {
                    'videoId': item['id'],
                    'viewCount': item['statistics'].get('viewCount', 0),
                    'likeCount': item['statistics'].get('likeCount', 0),
                    'commentCount': item['statistics'].get('commentCount', 0),
                    'duration': item['contentDetails'].get('duration', ''),
                    'definition': item['contentDetails'].get('definition', ''),  # 'hd' or 'sd'
                    'caption': item['contentDetails'].get('caption', 'false') == 'true',  # 자막 있는지 여부
                }
                all_video_details.append(video_detail)
            
            # API 할당량 고려하여 약간의 지연 추가
            if i + batch_size < len(video_ids):
                time.sleep(0.5)
                
        except HttpError as e:
            print(f"비디오 상세 정보 API 요청 중 오류 발생: {e}")
            if e.resp.status == 403:
                print("오류 403: 할당량 초과 또는 API 키/권한 문제일 수 있습니다.")
            # 오류 발생해도 지금까지 수집한 정보는 반환
            break
        except Exception as e:
            print(f"비디오 상세 정보 조회 중 알 수 없는 오류 발생: {e}")
            break
    
    print(f"총 {len(all_video_details)}개 비디오의 상세 정보를 조회했습니다.")
    return all_video_details


def get_channel_details(
    channel_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    channels.list API를 사용하여 채널의 상세 정보를 가져옵니다.
    
    Args:
        channel_ids (List[str]): 정보를 가져올 채널 ID 목록
        
    Returns:
        List[Dict[str, Any]]: 채널 상세 정보 목록
    """
    youtube_service = youtube_api.get_service()
    if not youtube_service:
        print("오류: YouTube API 서비스 초기화에 실패했습니다.")
        return []
    
    # 50개씩 나누어 처리 (API 제한)
    batch_size = 50
    all_channel_details = []
    
    for i in range(0, len(channel_ids), batch_size):
        batch_ids = channel_ids[i:i + batch_size]
        print(f"채널 상세 정보 조회 중: {i+1}~{min(i+batch_size, len(channel_ids))} / {len(channel_ids)}")
        
        try:
            request = youtube_service.channels().list(
                part="snippet,statistics,brandingSettings",
                id=",".join(batch_ids)
            )
            response = request.execute()
            
            for item in response.get("items", []):
                channel_detail = {
                    'channelId': item['id'],
                    'subscriberCount': item['statistics'].get('subscriberCount', 0),
                    'videoCount': item['statistics'].get('videoCount', 0),
                    'viewCount': item['statistics'].get('viewCount', 0),
                    'country': item['snippet'].get('country', ''),
                    'publishedAt': item['snippet'].get('publishedAt', ''),
                }
                
                # 채널 설명 및 키워드 (있는 경우만)
                if 'description' in item['snippet']:
                    channel_detail['description'] = item['snippet']['description']
                
                if 'keywords' in item.get('brandingSettings', {}).get('channel', {}):
                    channel_detail['keywords'] = item['brandingSettings']['channel'].get('keywords', '')
                
                all_channel_details.append(channel_detail)
            
            # API 할당량 고려하여 약간의 지연 추가
            if i + batch_size < len(channel_ids):
                time.sleep(0.5)
                
        except HttpError as e:
            print(f"채널 상세 정보 API 요청 중 오류 발생: {e}")
            if e.resp.status == 403:
                print("오류 403: 할당량 초과 또는 API 키/권한 문제일 수 있습니다.")
            # 오류 발생해도 지금까지 수집한 정보는 반환
            break
        except Exception as e:
            print(f"채널 상세 정보 조회 중 알 수 없는 오류 발생: {e}")
            break
    
    print(f"총 {len(all_channel_details)}개 채널의 상세 정보를 조회했습니다.")
    return all_channel_details


def search_and_get_details(
    query: str,
    max_results: int = 30,
    content_type: str = "video",
    include_details: bool = True
) -> pd.DataFrame:
    """
    키워드로 YouTube 검색을 수행하고, 선택적으로 상세 정보도 함께 가져옵니다.
    
    Args:
        query (str): 검색할 키워드 문자열
        max_results (int, optional): 반환할 최대 결과 수 (기본값: 30)
        content_type (str, optional): 검색할 콘텐츠 유형 (기본값: "video")
        include_details (bool, optional): 상세 정보도 함께 가져올지 여부 (기본값: True)
        
    Returns:
        pd.DataFrame: 검색 결과와 상세 정보가 병합된 DataFrame
    """
    # 1단계: 키워드로 검색 수행
    search_results = search_youtube(
        query=query,
        max_total_results=max_results,
        content_type=content_type
    )
    
    if not search_results:
        print("검색 결과가 없습니다.")
        return pd.DataFrame()
    
    # 2단계: 검색 결과를 DataFrame으로 변환
    df = pd.DataFrame(search_results)
    
    # 3단계: 상세 정보 조회 (선택 사항)
    if include_details:
        # 동영상 상세 정보 조회
        if 'video' in content_type and 'videoId' in df.columns:
            video_ids = df[df['kind'] == 'youtube#video']['videoId'].dropna().tolist()
            if video_ids:
                print("\n동영상 상세 정보 가져오는 중...")
                video_details = get_video_details(video_ids)
                if video_details:
                    df_video_details = pd.DataFrame(video_details)
                    # videoId를 기준으로 검색 결과와 상세 정보 병합
                    df = pd.merge(df, df_video_details, on='videoId', how='left')
        
        # 채널 상세 정보 조회
        if 'channel' in content_type and 'channelId' in df.columns:
            channel_ids = []
            # 채널 검색 결과의 channelId_item과 비디오 검색 결과의 channelId 모두 수집
            if 'channelId_item' in df.columns:
                channel_ids.extend(df[df['kind'] == 'youtube#channel']['channelId_item'].dropna().tolist())
            
            # 중복 제거 (비디오와 채널 검색에서 같은 채널이 나올 수 있음)
            channel_ids = list(set(channel_ids))
            
            if channel_ids:
                print("\n채널 상세 정보 가져오는 중...")
                channel_details = get_channel_details(channel_ids)
                if channel_details:
                    df_channel_details = pd.DataFrame(channel_details)
                    
                    # 채널인 경우 channelId_item을 channelId로 매핑하여 병합
                    if 'channelId_item' in df.columns:
                        # 임시로 channelId_item을 channelId로 복사
                        temp_df = df[df['kind'] == 'youtube#channel'].copy()
                        if not temp_df.empty:
                            temp_df['channelId'] = temp_df['channelId_item']
                            # 원래 DataFrame에서 해당 행들 제거
                            df = df[df['kind'] != 'youtube#channel']
                            # 변환된 행들 다시 추가
                            df = pd.concat([df, temp_df])
                    
                    # channelId를 기준으로 병합
                    df = pd.merge(df, df_channel_details, on='channelId', how='left', suffixes=('', '_channel'))
    
    # 4단계: 데이터 정리 (선택적)
    # 필요에 따라 열 이름 변경, 불필요한 열 제거, 데이터 타입 변환 등 수행 가능
    
    return df


# 테스트 코드
if __name__ == "__main__":
    # 테스트 검색 쿼리
    TEST_QUERY = "파이썬 프로그래밍 기초"
    
    print(f"검색어 '{TEST_QUERY}'로 YouTube 검색 및 상세 정보 수집 테스트")
    
    # 동영상만 검색하고 상세 정보도 함께 가져오기
    result_df = search_and_get_details(
        query=TEST_QUERY,
        max_results=10,  # 테스트용으로 적은 수만 검색
        content_type="video",
        include_details=True
    )
    
    if not result_df.empty:
        print("\n--- 검색 결과 및 상세 정보 (상위 5개) ---")
        display_cols = ['title', 'videoId', 'channelTitle', 'viewCount', 'publishedAt']
        print(result_df[display_cols].head(5))
        
        # 결과를 파일로 저장 (주석 해제하여 사용 가능)
        # result_df.to_csv("youtube_search_results.csv", index=False, encoding='utf-8-sig') 