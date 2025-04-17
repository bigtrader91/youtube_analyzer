# YouTube 콘텐츠 분석 시스템

YouTube 동영상 검색, 스크립트 및 댓글 추출, NLP 키워드 분석, 리포트 생성 기능을 제공하는 종합 콘텐츠 분석 시스템입니다.

## 주요 기능

- **YouTube 검색**: 키워드 기반 동영상 및 채널 검색
- **데이터 수집**: 스크립트 추출 및 댓글 수집
- **NLP 분석**: YAKE, RAKE, KeyBERT 알고리즘을 활용한 키워드 추출
- **리포트 생성**: 워드클라우드를 포함한 시각적 분석 리포트
- **웹 인터페이스**: Streamlit 기반 사용자 친화적 인터페이스

## 시작하기

### 필수 요구사항

- Python 3.7 이상
- YouTube Data API 키
- (선택) Google Ads API 개발자 토큰 및 OAuth 인증

### 설치 방법

1. 저장소 복제
```bash
git clone https://github.com/yourusername/youtube_analyzer.git
cd youtube_analyzer
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

4. (한국어 처리 지원을 위한 추가 설정)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

5. API 키 설정
   - `.env.example` 파일을 `.env`로 복사하고 필요한 API 키와 설정을 입력

### 실행 방법

#### 명령줄 인터페이스 (CLI)

```bash
# 키워드로 검색 및 전체 분석 수행
python -m src.main --query "검색어" --max_results 10

# 특정 비디오 ID로 분석
python -m src.main --mode analyze --video_ids VIDEO_ID1 VIDEO_ID2

# 도움말 보기
python -m src.main --help
```

#### 웹 인터페이스 (Streamlit)

```bash
streamlit run app.py
```
웹 브라우저가 자동으로 열리고 인터페이스가 표시됩니다.

## 모듈 구조

- **src/api_clients.py**: API 클라이언트 초기화 및 관리
- **src/youtube_searcher.py**: YouTube 검색 기능
- **src/transcript_extractor.py**: 동영상 스크립트 추출
- **src/comment_extractor.py**: 동영상 댓글 추출
- **src/nlp_processor.py**: 텍스트 전처리 및 키워드 추출
- **src/data_manager.py**: 데이터 저장 및 리포트 생성
- **src/main.py**: CLI 인터페이스
- **app.py**: Streamlit 웹 인터페이스

## API 설정 가이드

YouTube Data API 키를 얻기 위한 과정:

1. Google Cloud Console(https://console.cloud.google.com/)에서 새 프로젝트 생성
2. YouTube Data API v3 활성화
3. API 키 생성 및 제한 설정
4. .env 파일에 API 키 저장

자세한 설정 방법은 `docs/8. API 설정 가이드 (YouTube Data API & Google Ads API).md` 파일을 참조하세요.

## 사용 예시

### 웹 인터페이스 사용법

1. API 키 설정: 사이드바에 YouTube API 키 입력 및 초기화
2. YouTube 검색: 키워드 입력 및 검색 실행
3. 동영상 선택: 분석할 동영상 ID 선택
4. 데이터 수집: 스크립트 및 댓글 추출
5. 키워드 분석: NLP 방법 선택 및 키워드 추출
6. 리포트 생성: 종합 리포트 및 워드클라우드 생성

## 주의사항

- API 할당량: YouTube Data API는 일일 할당량이 제한되어 있습니다.
- 보안: API 키 및 인증 정보는 안전하게 관리하세요.
- 한국어 처리: 한국어 분석을 위해서는 konlpy와 관련 의존성 설치가 필요합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
