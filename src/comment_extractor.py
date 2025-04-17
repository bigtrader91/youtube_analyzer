"""
YouTube 동영상 댓글 추출 모듈 - YouTube Data API 활용
"""
import time
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
from googleapiclient.errors import HttpError

from src.api_clients import youtube_api


def get_comment_replies(
    youtube_service,
    parent_id: str,
    max_results: int = 100
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    주어진 부모 댓글 ID에 대한 답글을 가져옵니다.
    
    Args:
        youtube_service: YouTube API 서비스 객체
        parent_id (str): 부모 댓글 ID
        max_results (int, optional): 최대 가져올 답글 수 (기본값: 100)
        
    Returns:
        Tuple[List[Dict[str, Any]], bool]: (답글 데이터 리스트, 실패 여부)
    """
    replies_data = []
    failed = False
    next_page_token = None
    
    while len(replies_data) < max_results:
        try:
            current_max = min(100, max_results - len(replies_data))  # 페이지당 최대 100개
            if current_max <= 0:
                break
            
            request = youtube_service.comments().list(
                part="snippet",
                parentId=parent_id,
                textFormat="plainText",
                maxResults=current_max,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get("items", []):
                snippet = item['snippet']
                replies_data.append({
                    'commentId': item['id'],
                    'text': snippet['textDisplay'],
                    'author': snippet['authorDisplayName'],
                    'publishedAt': snippet['publishedAt'],
                    'updatedAt': snippet['updatedAt'],
                    'likeCount': snippet['likeCount']
                })
                if len(replies_data) >= max_results:
                    break
            
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
                
            # API 할당량 고려하여 약간의 지연 추가
            time.sleep(0.3)
            
        except HttpError as e:
            print(f"    답글 조회 중 오류 (Parent ID: {parent_id}): {e}")
            failed = True
            break
        except Exception as e:
            print(f"    답글 조회 중 알 수 없는 오류 (Parent ID: {parent_id}): {e}")
            failed = True
            break
    
    return replies_data, failed


def get_video_comments(
    video_id: str,
    max_total_comments: int = 100,
    fetch_replies: bool = True,
    max_replies_per_comment: int = 10,
    order: str = "relevance"
) -> List[Dict[str, Any]]:
    """
    YouTube Data API를 사용하여 동영상 댓글(및 답글)을 추출합니다.
    
    Args:
        video_id (str): 댓글을 추출할 YouTube 동영상 ID
        max_total_comments (int, optional): 수집할 총 최대 댓글 수 (최상위 댓글 기준) (기본값: 100)
        fetch_replies (bool, optional): 답글도 수집할지 여부 (기본값: True)
        max_replies_per_comment (int, optional): 각 댓글당 최대 답글 수 (기본값: 10)
        order (str, optional): 댓글 정렬 방식 (기본값: "relevance", 가능한 값: "relevance" 또는 "time")
        
    Returns:
        List[Dict[str, Any]]: 추출된 댓글 데이터
    """
    youtube_service = youtube_api.get_service()
    if not youtube_service:
        print("오류: YouTube API 서비스 초기화에 실패했습니다.")
        return []
    
    all_comments_data = []
    next_page_token = None
    
    print(f"비디오 ID '{video_id}'의 댓글 추출 시작...")
    
    while len(all_comments_data) < max_total_comments:
        try:
            # 최상위 댓글 요청
            request = youtube_service.commentThreads().list(
                part="snippet,replies" if fetch_replies else "snippet",
                videoId=video_id,
                textFormat="plainText",
                order=order,  # "relevance" 또는 "time"
                maxResults=min(100, max_total_comments - len(all_comments_data)),  # 페이지당 최대 100개
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get("items", []):
                top_comment_snippet = item['snippet']['topLevelComment']['snippet']
                comment_data = {
                    'commentId': item['snippet']['topLevelComment']['id'],
                    'text': top_comment_snippet['textDisplay'],
                    'author': top_comment_snippet['authorDisplayName'],
                    'publishedAt': top_comment_snippet['publishedAt'],
                    'updatedAt': top_comment_snippet['updatedAt'],
                    'likeCount': top_comment_snippet.get('likeCount', 0),
                    'totalReplyCount': item['snippet'].get('totalReplyCount', 0),
                    'replies': []
                }
                
                # 답글 수집 로직 (선택 사항)
                if fetch_replies and comment_data['totalReplyCount'] > 0:
                    print(f"  댓글 ID '{comment_data['commentId']}'의 답글({comment_data['totalReplyCount']}개) 수집 시도...")
                    replies_data, failed_replies = get_comment_replies(
                        youtube_service,
                        comment_data['commentId'],
                        min(comment_data['totalReplyCount'], max_replies_per_comment)  # 너무 많은 답글 제한
                    )
                    comment_data['replies'] = replies_data
                    if failed_replies:
                        print(f"    답글 수집 중 일부 오류 발생 (댓글 ID: {comment_data['commentId']})")
                
                all_comments_data.append(comment_data)
                if len(all_comments_data) >= max_total_comments:
                    break  # 목표 수 도달 시 중단
            
            next_page_token = response.get("nextPageToken")
            print(f"현재까지 수집된 최상위 댓글 수: {len(all_comments_data)}")
            
            if not next_page_token:
                print("더 이상 댓글 페이지가 없습니다.")
                break
            
            # API 할당량 고려 지연 추가
            time.sleep(0.5)
            
        except HttpError as e:
            print(f"API 요청 중 오류 발생: {e}")
            if e.resp.status == 403:
                if 'commentsDisabled' in str(e):
                    print("오류: 이 동영상에는 댓글 기능이 비활성화되어 있습니다.")
                else:
                    print("오류 403: 할당량 초과 또는 API 키/권한 문제일 수 있습니다.")
            break  # 오류 발생 시 중단
        except Exception as e:
            print(f"알 수 없는 오류 발생: {e}")
            break
    
    print(f"총 {len(all_comments_data)}개의 최상위 댓글 스레드를 수집했습니다.")
    return all_comments_data


def extract_all_comments_text(comments_data: List[Dict[str, Any]]) -> List[str]:
    """
    댓글 데이터에서 모든 텍스트(최상위 댓글 + 답글)를 추출합니다.
    
    Args:
        comments_data (List[Dict[str, Any]]): get_video_comments()로 얻은 댓글 데이터
        
    Returns:
        List[str]: 모든 댓글 텍스트 목록
    """
    all_texts = []
    
    for comment in comments_data:
        # 최상위 댓글 텍스트 추가
        all_texts.append(comment['text'])
        
        # 답글 텍스트 추가
        for reply in comment.get('replies', []):
            all_texts.append(reply['text'])
    
    return all_texts


def get_comments_for_videos(
    video_ids: List[str],
    max_comments_per_video: int = 50,
    fetch_replies: bool = True,
    max_replies_per_comment: int = 5,
    order: str = "relevance"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    여러 동영상에 대한 댓글을 수집합니다.
    
    Args:
        video_ids (List[str]): 댓글을 수집할 동영상 ID 목록
        max_comments_per_video (int, optional): 동영상당 최대 댓글 수 (기본값: 50)
        fetch_replies (bool, optional): 답글도 수집할지 여부 (기본값: True)
        max_replies_per_comment (int, optional): 각 댓글당 최대 답글 수 (기본값: 5)
        order (str, optional): 댓글 정렬 방식 (기본값: "relevance")
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: {video_id: comments_data, ...} 형태의 결과
    """
    results = {}
    failed_videos = []
    
    for i, video_id in enumerate(video_ids):
        print(f"\n[{i+1}/{len(video_ids)}] 비디오 ID '{video_id}'의 댓글 수집 중...")
        try:
            comments = get_video_comments(
                video_id=video_id,
                max_total_comments=max_comments_per_video,
                fetch_replies=fetch_replies,
                max_replies_per_comment=max_replies_per_comment,
                order=order
            )
            
            if comments:
                results[video_id] = comments
                print(f"비디오 ID '{video_id}': {len(comments)}개의 댓글 스레드 수집 완료")
            else:
                print(f"비디오 ID '{video_id}': 댓글이 없거나 수집 실패")
                failed_videos.append({'videoId': video_id, 'reason': '댓글 없음 또는 수집 실패'})
            
            # 여러 비디오 처리 시 API 할당량 고려하여 지연 추가
            if i < len(video_ids) - 1:
                print("다음 비디오 처리 전 잠시 대기 중...")
                time.sleep(2)
                
        except Exception as e:
            print(f"비디오 ID '{video_id}' 처리 중 예외 발생: {e}")
            failed_videos.append({'videoId': video_id, 'reason': f'Exception: {e}'})
    
    print(f"\n총 {len(results)}/{len(video_ids)} 비디오의 댓글 수집 완료")
    if failed_videos:
        print(f"{len(failed_videos)}개 비디오에서 댓글 수집 실패")
    
    return results


def comments_to_dataframe(comments_data: List[Dict[str, Any]], include_replies: bool = True) -> pd.DataFrame:
    """
    댓글 데이터를 DataFrame으로 변환합니다.
    
    Args:
        comments_data (List[Dict[str, Any]]): get_video_comments()로 얻은 댓글 데이터
        include_replies (bool, optional): 답글을 포함할지 여부 (기본값: True)
        
    Returns:
        pd.DataFrame: 댓글 데이터 DataFrame
    """
    if not comments_data:
        return pd.DataFrame()
    
    rows = []
    
    for comment in comments_data:
        # 최상위 댓글 행 추가
        rows.append({
            'commentId': comment['commentId'],
            'text': comment['text'],
            'author': comment['author'],
            'publishedAt': comment['publishedAt'],
            'likeCount': comment['likeCount'],
            'totalReplyCount': comment['totalReplyCount'],
            'isReply': False,
            'parentId': None
        })
        
        # 답글 행 추가 (선택 사항)
        if include_replies:
            for reply in comment.get('replies', []):
                rows.append({
                    'commentId': reply['commentId'],
                    'text': reply['text'],
                    'author': reply['author'],
                    'publishedAt': reply['publishedAt'],
                    'likeCount': reply.get('likeCount', 0),
                    'totalReplyCount': 0,  # 답글에는 답글이 없음
                    'isReply': True,
                    'parentId': comment['commentId']
                })
    
    return pd.DataFrame(rows)


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 동영상 ID (실제 예제)
    TEST_VIDEO_ID = "ogfYd705cRs"  # 예시 비디오 ID (실제 ID로 변경 필요)
    
    print(f"YouTube 동영상 ID '{TEST_VIDEO_ID}'의 댓글 추출 테스트")
    
    # 단일 비디오 댓글 추출
    comments_result = get_video_comments(
        video_id=TEST_VIDEO_ID,
        max_total_comments=20,  # 테스트용으로 적은 수만 추출
        fetch_replies=True,
        max_replies_per_comment=5
    )
    
    if comments_result:
        # 댓글을 DataFrame으로 변환하여 출력
        df_comments = comments_to_dataframe(comments_result)
        
        print("\n--- 추출된 댓글 (상위 5개) ---")
        print(df_comments[['commentId', 'text', 'author', 'likeCount', 'isReply']].head(5))
        
        # 모든 댓글 텍스트 추출 예시
        all_comment_texts = extract_all_comments_text(comments_result)
        
        print(f"\n총 {len(all_comment_texts)}개의 댓글/답글 텍스트 추출")
        print("첫 번째 댓글 텍스트 미리보기:")
        if all_comment_texts:
            preview = all_comment_texts[0][:200] + "..." if len(all_comment_texts[0]) > 200 else all_comment_texts[0]
            print(preview)
        
        # 결과를 파일로 저장 (주석 해제하여 사용 가능)
        # df_comments.to_csv("video_comments.csv", index=False, encoding='utf-8-sig') 