"""
API 클라이언트 초기화 모듈 - YouTube Data API 및 Google Ads API 클라이언트를 초기화합니다.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google Ads API는 선택적 기능으로 처리
try:
    from google.ads.googleads.client import GoogleAdsClient
    from google.ads.googleads.errors import GoogleAdsException
    GOOGLE_ADS_AVAILABLE = True
except ImportError:
    GOOGLE_ADS_AVAILABLE = False
    print("경고: Google Ads API 패키지가 설치되지 않았습니다. Google Ads 관련 기능은 사용할 수 없습니다.")

try:
    from src import config
except ImportError:
    print("경고: config 모듈을 찾을 수 없습니다. 환경 변수를 직접 사용합니다.")
    # config 모듈이 없을 경우 임시 대체 클래스 정의
    class TempConfig:
        YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
        GOOGLE_ADS_YAML_PATH = os.environ.get("GOOGLE_ADS_YAML_PATH", "")
        GADS_CLIENT_ID = os.environ.get("GOOGLE_ADS_CLIENT_ID", "")
        GADS_CLIENT_SECRET = os.environ.get("GOOGLE_ADS_CLIENT_SECRET", "")
        GADS_DEVELOPER_TOKEN = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN", "")
        GADS_REFRESH_TOKEN = os.environ.get("GOOGLE_ADS_REFRESH_TOKEN", "")
        GADS_LOGIN_CUSTOMER_ID = os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "")
    
    config = TempConfig()


class YouTubeAPIClient:
    """YouTube Data API 클라이언트 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        YouTube Data API 클라이언트를 초기화합니다.
        
        Args:
            api_key (str, optional): YouTube Data API 키
                                    미지정시 환경 변수에서 로드
        """
        self.api_key = api_key or config.YOUTUBE_API_KEY
        self.service = None
        
    def initialize(self) -> bool:
        """YouTube API 서비스를 초기화합니다."""
        if not self.api_key:
            print("오류: YouTube API 키가 설정되지 않았습니다.")
            return False
        
        try:
            self.service = build('youtube', 'v3', developerKey=self.api_key)
            return True
        except HttpError as e:
            print(f"오류: YouTube API 서비스 초기화 실패: {e}")
            return False
    
    def get_service(self):
        """초기화된 YouTube API 서비스 객체를 반환합니다."""
        if not self.service:
            self.initialize()
        return self.service


class GoogleAdsAPIClient:
    """Google Ads API 클라이언트 클래스"""
    
    def __init__(
        self,
        yaml_path: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        developer_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        login_customer_id: Optional[str] = None
    ):
        """
        Google Ads API 클라이언트를 초기화합니다.
        
        Args:
            yaml_path (str, optional): Google Ads YAML 설정 파일 경로
            client_id (str, optional): OAuth 클라이언트 ID
            client_secret (str, optional): OAuth 클라이언트 시크릿
            developer_token (str, optional): 개발자 토큰
            refresh_token (str, optional): OAuth 리프레시 토큰
            login_customer_id (str, optional): 관리자 계정(MCC) ID
        """
        # 기본값 설정 (yaml_path 우선, 미지정시 환경 변수 사용)
        self.yaml_path = yaml_path or config.GOOGLE_ADS_YAML_PATH
        self.client_id = client_id or config.GADS_CLIENT_ID
        self.client_secret = client_secret or config.GADS_CLIENT_SECRET
        self.developer_token = developer_token or config.GADS_DEVELOPER_TOKEN
        self.refresh_token = refresh_token or config.GADS_REFRESH_TOKEN
        self.login_customer_id = login_customer_id or config.GADS_LOGIN_CUSTOMER_ID
        
        self.client = None
    
    def initialize(self) -> bool:
        """Google Ads API 클라이언트를 초기화합니다."""
        if not GOOGLE_ADS_AVAILABLE:
            print("오류: Google Ads API 패키지가 설치되지 않았습니다.")
            return False
            
        try:
            # YAML 파일이 있으면 파일 기반 초기화
            if self.yaml_path and os.path.exists(self.yaml_path):
                self.client = GoogleAdsClient.load_from_storage(self.yaml_path)
                return True
            
            # 환경 변수 기반 초기화
            if all([self.client_id, self.client_secret, self.developer_token, self.refresh_token]):
                config_dict = {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "developer_token": self.developer_token,
                    "refresh_token": self.refresh_token
                }
                
                if self.login_customer_id:
                    config_dict["login_customer_id"] = self.login_customer_id.replace("-", "")
                
                self.client = GoogleAdsClient.load_from_dict(config_dict)
                return True
            
            # 인증 정보 부족
            print("오류: Google Ads API 인증 정보가 부족합니다.")
            print("google-ads.yaml 파일이나 환경 변수를 통해 필요한 인증 정보를 제공해주세요.")
            return False
            
        except Exception as e:
            print(f"오류: Google Ads API 클라이언트 초기화 중 예외 발생: {e}")
            return False
    
    def get_client(self):
        """초기화된 Google Ads API 클라이언트 객체를 반환합니다."""
        if not self.client:
            self.initialize()
        return self.client


# 싱글톤 인스턴스들
youtube_api = YouTubeAPIClient()
googleads_api = GoogleAdsAPIClient()


def initialize_all_clients() -> Tuple[bool, bool]:
    """모든 API 클라이언트를 초기화하고 성공 여부를 반환합니다."""
    youtube_success = youtube_api.initialize()
    
    # Google Ads API가 사용 가능한 경우에만 초기화 시도
    if GOOGLE_ADS_AVAILABLE:
        googleads_success = googleads_api.initialize()
    else:
        googleads_success = False
    
    return youtube_success, googleads_success 