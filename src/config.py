"""
설정 모듈 - 환경 변수 및 설정을 관리합니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우)
env_path = Path(__file__).parent.parent / '.env'
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# YouTube Data API 설정
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# Google Ads API 설정 (선택 사항)
GOOGLE_ADS_YAML_PATH = os.environ.get("GOOGLE_ADS_YAML_PATH", "")
GADS_CLIENT_ID = os.environ.get("GOOGLE_ADS_CLIENT_ID", "")
GADS_CLIENT_SECRET = os.environ.get("GOOGLE_ADS_CLIENT_SECRET", "")
GADS_DEVELOPER_TOKEN = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN", "")
GADS_REFRESH_TOKEN = os.environ.get("GOOGLE_ADS_REFRESH_TOKEN", "")
GADS_LOGIN_CUSTOMER_ID = os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "")

# 기타 설정
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "ko")
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "./data")

# 필요한 디렉토리 확인
def ensure_directories():
    """필요한 데이터 디렉토리가 있는지 확인하고 없으면 생성합니다."""
    data_dir = Path(DEFAULT_DATA_DIR)
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"데이터 디렉토리 생성 완료: {data_dir}")
        except Exception as e:
            print(f"데이터 디렉토리 생성 실패: {e}")
    
    return data_dir.exists()

# 설정 유효성 검사
def validate_settings():
    """설정 값의 유효성을 검사합니다."""
    # YouTube API 키 검사
    if not YOUTUBE_API_KEY:
        print("경고: YouTube API 키가 설정되지 않았습니다.")
        return False
    
    return True 