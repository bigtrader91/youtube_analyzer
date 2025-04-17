"""
Google Keyword Planner 통합 모듈 - Google Ads API를 활용하여 키워드 아이디어 생성 및 측정항목 조회
"""
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import time

from src.api_clients import googleads_api


def get_keyword_ideas(
    customer_id: str,
    seed_keywords: List[str],
    language_id: str = "1012",  # 기본 한국어
    location_ids: List[str] = ["2410"],  # 기본 대한민국
    max_results: int = 1000,
) -> List[str]:
    """
    KeywordPlanIdeaService를 사용하여 시드 키워드를 기반으로 키워드 아이디어를 생성합니다.
    
    Args:
        customer_id (str): Google Ads 고객 ID
        seed_keywords (List[str]): 아이디어 생성을 위한 시드 키워드 목록
        language_id (str, optional): 언어 ID (기본값: "1012" - 한국어)
        location_ids (List[str], optional): 지역 ID 목록 (기본값: ["2410"] - 대한민국)
        max_results (int, optional): 최대 결과 수 (기본값: 1000)
        
    Returns:
        List[str]: 생성된 키워드 아이디어 목록
    """
    client = googleads_api.get_client()
    if not client:
        print("오류: Google Ads API 클라이언트 초기화에 실패했습니다.")
        return []
    
    # 한 번에 너무 많은 시드 키워드를 보내면 API 할당량 문제가 발생할 수 있음
    if len(seed_keywords) > 10:
        print(f"경고: 시드 키워드가 너무 많습니다({len(seed_keywords)}개). 처음 10개만 사용합니다.")
        seed_keywords = seed_keywords[:10]
    
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
    keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
    
    # 지역 및 언어 리소스 이름 생성
    location_rns = [f"geoTargetConstants/{loc_id}" for loc_id in location_ids]
    language_rn = f"languageConstants/{language_id}"
    
    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = customer_id
    request.language = language_rn
    request.geo_target_constants = location_rns
    request.keyword_plan_network = keyword_plan_network
    request.keyword_seed.keywords.extend(seed_keywords)
    
    # 필요시 URL Seed 등 다른 옵션 추가 가능
    # request.url_seed.url = "YOUR_URL"
    
    ideas = []
    
    try:
        response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        
        for idx, result in enumerate(response.results):
            if idx >= max_results:
                break
            ideas.append(result.text)
            
        print(f"총 {len(ideas)}개의 키워드 아이디어를 생성했습니다.")
        return ideas
    
    except GoogleAdsException as ex:
        print(f"키워드 아이디어 생성 중 Google Ads API 오류 발생:")
        for error in ex.failure.errors:
            print(f"\t{error.error_code.message}: {error.message}")
            
        if any(error.error_code.quota_error for error in ex.failure.errors):
            print("API 할당량(Quota) 초과 오류가 발생했습니다. 나중에 다시 시도하세요.")
            
        return []
    
    except Exception as e:
        print(f"키워드 아이디어 생성 중 예외 발생: {e}")
        return []


def get_historical_metrics(
    customer_id: str,
    keywords: List[str],
    language_id: str = "1012",  # 기본 한국어
    location_ids: List[str] = ["2410"],  # 기본 대한민국
    batch_size: int = 500,  # API 할당량을 고려한 배치 크기
) -> List[Dict[str, Any]]:
    """
    KeywordPlanHistoricalMetricsService를 사용하여 키워드의 과거 측정항목을 조회합니다.
    
    Args:
        customer_id (str): Google Ads 고객 ID
        keywords (List[str]): 측정항목을 조회할 키워드 목록
        language_id (str, optional): 언어 ID (기본값: "1012" - 한국어)
        location_ids (List[str], optional): 지역 ID 목록 (기본값: ["2410"] - 대한민국)
        batch_size (int, optional): 한 번에 처리할 키워드 수 (기본값: 500)
        
    Returns:
        List[Dict[str, Any]]: 키워드별 측정항목 데이터를 포함하는 사전 목록
    """
    client = googleads_api.get_client()
    if not client:
        print("오류: Google Ads API 클라이언트 초기화에 실패했습니다.")
        return []
    
    keyword_plan_historical_metrics_service = client.get_service("KeywordPlanHistoricalMetricsService")
    location_rns = [f"geoTargetConstants/{loc_id}" for loc_id in location_ids]
    language_rn = f"languageConstants/{language_id}"
    
    all_keyword_metrics_data = []
    
    # 대용량 키워드 목록 처리를 위한 배치 처리
    for i in range(0, len(keywords), batch_size):
        batch_keywords = keywords[i:i + batch_size]
        print(f"배치 처리 중: {i+1}~{min(i+batch_size, len(keywords))} / {len(keywords)} 키워드")
        
        request = client.get_type("GetKeywordPlanHistoricalMetricsRequest")
        request.customer_id = customer_id
        request.keywords.extend(batch_keywords)
        request.language = language_rn
        request.geo_target_constants = location_rns
        request.historical_metrics_options.include_average_cpc = False  # CPC 정보는 이번 목표에서 제외
        
        try:
            response = keyword_plan_historical_metrics_service.get_keyword_plan_historical_metrics(request=request)
            
            competition_enum = client.enums.KeywordPlanCompetitionLevelEnum
            for result in response.results:
                metrics = result.keyword_metrics
                competition_level = competition_enum(metrics.competition).name if metrics.competition else 'UNKNOWN'
                
                keyword_metrics_data = {
                    'keyword': result.search_query,
                    'avg_monthly_searches': metrics.avg_monthly_searches if metrics.HasField('avg_monthly_searches') else 0,
                    'competition': competition_level
                }
                
                # 월별 검색량이 있을 경우 추가 (지난 12개월)
                if metrics.monthly_search_volumes:
                    month_year_data = {}
                    for monthly_data in metrics.monthly_search_volumes:
                        month_name = f"{monthly_data.month.name}_{monthly_data.year}"
                        month_year_data[month_name] = monthly_data.monthly_searches
                    keyword_metrics_data['monthly_searches'] = month_year_data
                
                all_keyword_metrics_data.append(keyword_metrics_data)
            
            # API 할당량 고려하여 배치 간 지연 추가
            if i + batch_size < len(keywords):
                print("API 할당량 보호를 위해 잠시 대기 중...")
                time.sleep(2)
                
        except GoogleAdsException as ex:
            print(f"과거 측정항목 조회 중 Google Ads API 오류 발생:")
            for error in ex.failure.errors:
                print(f"\t{error.error_code.message}: {error.message}")
                
            if any(error.error_code.quota_error for error in ex.failure.errors):
                print("API 할당량(Quota) 초과 오류가 발생했습니다. 나중에 다시 시도하세요.")
                
            # 나머지 배치는 건너뛰고 지금까지 수집된 데이터 반환
            break
            
        except Exception as e:
            print(f"과거 측정항목 조회 중 예외 발생: {e}")
            # 현재 배치는 건너뛰고 다음 배치로 진행
            continue
    
    print(f"총 {len(all_keyword_metrics_data)}개 키워드의 측정항목을 조회했습니다.")
    return all_keyword_metrics_data


def analyze_keywords(
    customer_id: str,
    seed_keywords: List[str],
    language_id: str = "1012",
    location_ids: List[str] = ["2410"],
    include_seed_metrics: bool = True,
    max_ideas: int = 500,
) -> pd.DataFrame:
    """
    시드 키워드를 분석하여 관련 키워드 아이디어와 측정항목을 생성합니다.
    
    Args:
        customer_id (str): Google Ads 고객 ID
        seed_keywords (List[str]): 아이디어 생성을 위한 시드 키워드 목록
        language_id (str, optional): 언어 ID (기본값: "1012" - 한국어)
        location_ids (List[str], optional): 지역 ID 목록 (기본값: ["2410"] - 대한민국)
        include_seed_metrics (bool, optional): 시드 키워드 자체에 대한 측정항목도 조회할지 여부
        max_ideas (int, optional): 생성할 최대 키워드 아이디어 수 (기본값: 500)
        
    Returns:
        pd.DataFrame: 키워드와 측정항목 데이터를 포함하는 Pandas DataFrame
    """
    # 1단계: 키워드 아이디어 생성
    print("\n1. 키워드 아이디어 생성 중...")
    generated_ideas = get_keyword_ideas(
        customer_id=customer_id,
        seed_keywords=seed_keywords,
        language_id=language_id,
        location_ids=location_ids,
        max_results=max_ideas
    )
    
    if not generated_ideas:
        print("키워드 아이디어 생성에 실패했습니다.")
        return pd.DataFrame()
    
    # 2단계: 측정항목 조회 대상 키워드 준비
    all_keywords = []
    
    # 시드 키워드 측정항목도 함께 조회할 경우
    if include_seed_metrics:
        all_keywords.extend(seed_keywords)
    
    # 중복 제거를 위해 set 사용
    all_keywords.extend(generated_ideas)
    unique_keywords = list(set(all_keywords))
    
    # 3단계: 측정항목 조회
    print("\n2. 키워드 측정항목 조회 중...")
    metrics_data = get_historical_metrics(
        customer_id=customer_id,
        keywords=unique_keywords,
        language_id=language_id,
        location_ids=location_ids
    )
    
    if not metrics_data:
        print("키워드 측정항목 조회에 실패했습니다.")
        return pd.DataFrame()
    
    # 4단계: DataFrame 생성 및 데이터 정렬
    df = pd.DataFrame(metrics_data)
    if not df.empty:
        # '시드 키워드 여부' 열 추가
        df['is_seed'] = df['keyword'].apply(lambda x: x in seed_keywords)
        
        # 검색량 기준 내림차순 정렬
        df = df.sort_values(by=['is_seed', 'avg_monthly_searches'], ascending=[False, False])
        
        # 경쟁도를 숫자로 변환하여 정렬 가능하게 함
        competition_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'UNKNOWN': 0}
        df['competition_score'] = df['competition'].map(competition_map)
        
        # 가독성을 위해 열 순서 조정
        columns = ['keyword', 'is_seed', 'avg_monthly_searches', 'competition', 'competition_score']
        if 'monthly_searches' in df.columns:
            columns.append('monthly_searches')
        
        df = df[columns]
    
    return df


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 예시 값 (실제 사용 시 변경 필요)
    TEST_CUSTOMER_ID = "123-456-7890"  # 실제 Google Ads 고객 ID로 변경 필요
    TEST_SEED_KEYWORDS = ["파이썬 기초", "파이썬 독학", "파이썬 개발자"]
    
    print("Google Keyword Planner 분석 테스트")
    print(f"시드 키워드: {TEST_SEED_KEYWORDS}")
    
    # 전체 분석 워크플로우 테스트
    result_df = analyze_keywords(
        customer_id=TEST_CUSTOMER_ID,
        seed_keywords=TEST_SEED_KEYWORDS,
        max_ideas=20  # 테스트용으로 적은 수만 생성
    )
    
    if not result_df.empty:
        print("\n--- 분석 결과 (상위 10개 키워드) ---")
        print(result_df.head(10))
        
        # 결과를 파일로 저장 (주석 해제하여 사용 가능)
        # result_df.to_csv("keyword_analysis_results.csv", index=False, encoding='utf-8-sig') 