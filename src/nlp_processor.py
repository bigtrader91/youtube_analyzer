"""
텍스트 데이터 전처리 및 NLP 키워드 추출 모듈 - YAKE, RAKE, KeyBERT 등 활용
"""
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import Counter

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import yake
from rake_nltk import Rake
from keybert import KeyBERT

# konlpy 임포트 (한국어 처리용)
try:
    from konlpy.tag import Okt
    has_konlpy = True
except ImportError:
    print("경고: konlpy 패키지를 찾을 수 없습니다. 한국어 처리 기능이 제한됩니다.")
    has_konlpy = False

# NLTK 리소스 확인 및 다운로드
def ensure_nltk_resources():
    """필요한 NLTK 리소스가 있는지 확인하고 없으면 다운로드합니다."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK punkt 다운로드 중...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK stopwords 다운로드 중...")
        nltk.download('stopwords', quiet=True)

# 전처리 관련 함수들
class TextPreprocessor:
    """텍스트 전처리를 위한 클래스"""
    
    def __init__(self, language='ko'):
        """
        TextPreprocessor 초기화
        
        Args:
            language (str): 텍스트 언어 코드 ('ko': 한국어, 'en': 영어)
        """
        ensure_nltk_resources()
        self.language = language
        
        # 불용어 사전 설정
        self.stopwords = set()
        if language == 'en':
            self.stopwords = set(stopwords.words('english'))
        elif language == 'ko':
            # 한국어 불용어 직접 정의 (확장 가능)
            korean_stopwords = set([
                '이', '그', '저', '것', '수', '이런', '저런', '그런', '한', '두', '이번', '저번', '그리고',
                '하지만', '그러나', '그래서', '그럼', '그렇게', '이렇게', '저렇게', '때문에', '까지', '부터',
                '있다', '하다', '되다', '것이다', '등', '이다', '같다', '경우', '통해', '따라', '위해',
                '때', '중', '더', '또', '또는', '및', '그리고', '여러', '어떤', '이미'
            ])
            self.stopwords = korean_stopwords
        
        # 한국어 처리를 위한 Okt 초기화 (있는 경우)
        self.okt = None
        if language == 'ko' and has_konlpy:
            try:
                self.okt = Okt()
                print("한국어 형태소 분석기(Okt) 초기화 완료")
            except Exception as e:
                print(f"Okt 초기화 오류: {e}")
    
    def remove_url(self, text):
        """URL 제거"""
        return re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    
    def remove_emoji(self, text):
        """이모지 제거"""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r' ', text)
    
    def remove_special_chars(self, text):
        """특수문자 제거 (알파벳, 숫자, 한글, 공백 유지)"""
        if self.language == 'ko':
            return re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    def normalize_whitespace(self, text):
        """여러 공백을 하나로 정규화"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def preprocess_text(self, text):
        """텍스트 전처리 실행"""
        if not isinstance(text, str):
            return ""
        
        # 기본 전처리 (언어 공통)
        text = self.remove_url(text)
        text = self.remove_emoji(text)
        
        # 영어 특화 전처리
        if self.language == 'en':
            text = text.lower()  # 소문자화 (영어만)
        
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        
        # 언어별 토큰화 및 불용어 제거
        if self.language == 'en':
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word.lower() not in self.stopwords and len(word) > 1]
            return " ".join(filtered_tokens)
        
        elif self.language == 'ko':
            if self.okt:
                # 한국어 형태소 분석 및 명사 추출
                try:
                    # 방법 1: 명사만 추출
                    nouns = self.okt.nouns(text)
                    filtered_nouns = [noun for noun in nouns if noun not in self.stopwords and len(noun) > 1]
                    
                    # 방법 2: 형태소 분석 후 명사/동사/형용사 등 필요한 품사만 선택적으로 추출
                    morphs = self.okt.pos(text)
                    filtered_morphs = []
                    for word, pos in morphs:
                        if (pos in ['Noun', 'Verb', 'Adjective'] and 
                            word not in self.stopwords and 
                            len(word) > 1):
                            filtered_morphs.append(word)
                    
                    # 두 방법 결과 합치기 (선택적)
                    combined = list(set(filtered_nouns + filtered_morphs))
                    return " ".join(combined)
                except Exception as e:
                    print(f"한국어 처리 오류: {e}")
                    # 오류 발생 시 원본 텍스트 반환 (최소한의 전처리만 적용)
                    return text
            else:
                # konlpy 없을 경우 단순 공백 기준 토큰화 및 불용어 제거
                tokens = text.split()
                filtered_tokens = [word for word in tokens if word not in self.stopwords and len(word) > 1]
                return " ".join(filtered_tokens)
        
        return text  # 그 외 언어는 기본 전처리만 적용
    
    def batch_preprocess(self, texts):
        """여러 텍스트 일괄 전처리"""
        if isinstance(texts, list):
            return [self.preprocess_text(text) for text in texts]
        return self.preprocess_text(texts)


# 키워드 추출 관련 함수들
class KeywordExtractor:
    """다양한 알고리즘으로 키워드를 추출하는 클래스"""
    
    def __init__(self, language='ko'):
        """
        KeywordExtractor 초기화
        
        Args:
            language (str): 텍스트 언어 코드 ('ko': 한국어, 'en': 영어)
        """
        self.language = language
        self.preprocessor = TextPreprocessor(language)
        
        # KeyBERT 모델은 최초 사용 시 초기화 (heavy한 모델이라서 필요할 때 초기화)
        self.keybert_model = None
    
    def _initialize_keybert(self):
        """KeyBERT 모델 초기화"""
        try:
            # 언어에 맞는 모델 선택
            if self.language == 'ko':
                # 한국어 모델 (설치되어 있다면)
                try:
                    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'  # 다국어 모델
                    self.keybert_model = KeyBERT(model=model_name)
                    print(f"KeyBERT 초기화 완료: {model_name}")
                except Exception as e:
                    print(f"한국어 KeyBERT 모델 초기화 오류: {e}")
                    # 오류 시 기본 다국어 모델 시도
                    self.keybert_model = KeyBERT()
            else:
                # 영어 또는 기타 언어
                self.keybert_model = KeyBERT()
        except Exception as e:
            print(f"KeyBERT 초기화 오류: {e}")
            self.keybert_model = None
    
    def extract_yake_keywords(self, text, top_n=10):
        """YAKE 알고리즘으로 키워드 추출"""
        try:
            # 언어 코드 맵핑 (YAKE에서 사용하는 코드와 일치시킴)
            lang_code = 'en' if self.language == 'en' else 'ko'
            
            # 단일 키워드(n=1)와 복합 키워드(n=2) 모두 추출
            kw_extractor = yake.KeywordExtractor(
                lan=lang_code, 
                n=1,  # unigram
                dedupLim=0.9, 
                dedupFunc='seqm', 
                windowsSize=2,
                top=top_n
            )
            keywords = kw_extractor.extract_keywords(text)
            
            # 복합 키워드(2단어) 추출
            kw_extractor_bigram = yake.KeywordExtractor(
                lan=lang_code, 
                n=2,  # bigram 
                dedupLim=0.9, 
                dedupFunc='seqm', 
                windowsSize=2,
                top=top_n//2  # 단일 키워드의 절반만 추출
            )
            bigram_keywords = kw_extractor_bigram.extract_keywords(text)
            
            # 결과 병합 (YAKE는 점수가 낮을수록 중요한 키워드)
            all_keywords = keywords + bigram_keywords
            all_keywords.sort(key=lambda x: x[1])  # 점수 기준 정렬
            return all_keywords[:top_n]  # 상위 N개 반환
            
        except Exception as e:
            print(f"YAKE 오류: {e}")
            return []
    
    def extract_rake_keywords(self, text, top_n=10):
        """RAKE 알고리즘으로 키워드 추출"""
        try:
            # 한국어의 경우 자체 정의 불용어 사용
            if self.language == 'ko':
                stopwords_list = list(self.preprocessor.stopwords)
                r = Rake(stopwords=stopwords_list, include_repeated_phrases=False)
            else:
                r = Rake(language='english', include_repeated_phrases=False)
            
            r.extract_keywords_from_text(text)
            keywords = r.get_ranked_phrases_with_scores()
            
            # 결과 형식 통일 [(keyword, score), ...] - RAKE는 (score, keyword) 형식임을 주의
            formatted_keywords = [(phrase, score) for score, phrase in keywords[:top_n]]
            return formatted_keywords
            
        except Exception as e:
            print(f"RAKE 오류: {e}")
            return []
    
    def extract_keybert_keywords(self, text, top_n=10):
        """KeyBERT 알고리즘으로 키워드 추출"""
        try:
            # 최초 사용 시 KeyBERT 모델 초기화
            if self.keybert_model is None:
                self._initialize_keybert()
            
            if self.keybert_model is None:
                print("KeyBERT 모델을 초기화할 수 없습니다.")
                return []
            
            # KeyBERT는 원본 텍스트를 사용 (내부적으로 임베딩 및 처리 수행)
            keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),  # 1~2 단어 구문 추출
                stop_words=list(self.preprocessor.stopwords),
                use_maxsum=True,  # 결과 다양성 향상
                nr_candidates=20,  # 후보 키워드 수
                top_n=top_n
            )
            
            return keywords  # [(keyword, score), ...]
            
        except Exception as e:
            print(f"KeyBERT 오류: {e}")
            return []
    
    def extract_keywords(self, text, methods=None, top_n=10, preprocess=True):
        """
        여러 알고리즘으로 키워드 추출
        
        Args:
            text (str): 키워드를 추출할 텍스트
            methods (List[str], optional): 사용할 추출 방법 리스트 (기본값: ['yake', 'rake', 'keybert'])
            top_n (int, optional): 각 방법별로 추출할 상위 키워드 수 (기본값: 10)
            preprocess (bool, optional): 전처리 수행 여부 (기본값: True)
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: 방법별 키워드 및 점수
        """
        if methods is None:
            methods = ['yake', 'rake', 'keybert']
        
        # 전처리
        processed_text = self.preprocessor.preprocess_text(text) if preprocess else text
        results = {}
        
        # 각 알고리즘으로 키워드 추출
        start_time = time.time()
        
        if 'yake' in methods:
            print("YAKE로 키워드 추출 중...")
            results['yake'] = self.extract_yake_keywords(processed_text, top_n)
        
        if 'rake' in methods:
            print("RAKE로 키워드 추출 중...")
            results['rake'] = self.extract_rake_keywords(processed_text, top_n)
        
        if 'keybert' in methods:
            print("KeyBERT로 키워드 추출 중...")
            # KeyBERT는 원본 텍스트를 사용하거나 최소한의 전처리만 적용
            minimal_processed = self.preprocessor.remove_url(text)
            minimal_processed = self.preprocessor.remove_emoji(minimal_processed)
            minimal_processed = self.preprocessor.normalize_whitespace(minimal_processed)
            results['keybert'] = self.extract_keybert_keywords(minimal_processed, top_n)
        
        print(f"키워드 추출 완료 (소요 시간: {time.time()-start_time:.2f}초)")
        return results
    
    def combine_keywords(self, keyword_results, top_n=15):
        """
        여러 알고리즘의 키워드 결과를 결합하여 가중 키워드 추출
        
        Args:
            keyword_results (Dict[str, List[Tuple[str, float]]]): extract_keywords() 결과
            top_n (int, optional): 반환할 통합 키워드 수 (기본값: 15)
            
        Returns:
            List[Dict[str, Any]]: 통합된 키워드 목록 (점수, 순위, 출처 알고리즘 포함)
        """
        all_keywords = []
        
        # 각 알고리즘별 키워드를 순위와 함께 기록
        for method, keywords in keyword_results.items():
            for rank, (keyword, score) in enumerate(keywords):
                # 점수 정규화 (방법에 따라 다름)
                normalized_score = 0
                
                if method == 'yake':
                    # YAKE는 점수가 낮을수록 중요 (0에 가까울수록 좋음)
                    # 1에서 빼서 높을수록 중요하게 변환
                    normalized_score = 1 - min(score, 1.0)
                elif method == 'rake':
                    # RAKE는 점수가 높을수록 중요
                    # 일반적으로 점수 범위가 크므로 log 스케일로 조정 가능
                    normalized_score = min(score / 10.0, 1.0)  # 10으로 나누고 1 제한
                elif method == 'keybert':
                    # KeyBERT는 0~1 사이 유사도 점수로, 이미 정규화되어 있음
                    normalized_score = score
                
                all_keywords.append({
                    'keyword': keyword,
                    'original_score': score,
                    'normalized_score': normalized_score,
                    'rank': rank + 1,  # 1부터 시작하는 순위
                    'method': method
                })
        
        # 키워드별로 그룹화하여 점수 합산
        keyword_groups = {}
        for item in all_keywords:
            keyword = item['keyword'].lower()  # 대소문자 구분 없이
            if keyword not in keyword_groups:
                keyword_groups[keyword] = {
                    'keyword': item['keyword'],
                    'total_score': item['normalized_score'],
                    'count': 1,
                    'methods': [item['method']],
                    'ranks': [item['rank']]
                }
            else:
                group = keyword_groups[keyword]
                group['total_score'] += item['normalized_score']
                group['count'] += 1
                group['methods'].append(item['method'])
                group['ranks'].append(item['rank'])
        
        # 최종 점수 계산 및 정렬
        final_keywords = []
        for keyword, data in keyword_groups.items():
            # 평균 점수와 등장 횟수를 모두 고려
            avg_score = data['total_score'] / data['count']
            avg_rank = sum(data['ranks']) / len(data['ranks'])
            
            # 여러 방법에서 나올수록 보너스 점수
            method_bonus = min(len(set(data['methods'])) / len(keyword_results), 1.0)
            
            final_score = avg_score * (0.7 + 0.3 * method_bonus)
            
            # 메서드 리스트를 문자열로 변환 (앱에서 사용하기 쉽게)
            methods_str = ", ".join(list(set(data['methods'])))
            
            final_keywords.append({
                'keyword': data['keyword'],
                'score': round(final_score, 4),
                'avg_rank': round(avg_rank, 2),
                'count': data['count'],
                'methods': methods_str  # 문자열로 변환
            })
        
        # 점수 기준 내림차순 정렬 후 상위 N개 반환
        final_keywords.sort(key=lambda x: x['score'], reverse=True)
        return final_keywords[:top_n]
    
    def keywords_to_dataframe(self, keyword_results):
        """
        키워드 결과를 DataFrame으로 변환
        
        Args:
            keyword_results (Dict[str, List[Tuple[str, float]]]): extract_keywords() 결과
            
        Returns:
            pd.DataFrame: 키워드 결과 DataFrame
        """
        rows = []
        
        for method, keywords in keyword_results.items():
            for rank, (keyword, score) in enumerate(keywords, 1):
                rows.append({
                    'keyword': keyword,
                    'score': score,
                    'rank': rank,
                    'method': method
                })
        
        return pd.DataFrame(rows)


# 실제 사용 예시 (테스트 코드)
if __name__ == "__main__":
    # 테스트 텍스트 (영어 및 한국어)
    test_text_en = """
    Welcome to YouTube keyword analysis! This tool helps content creators optimize their videos.
    We analyze transcripts, comments, and metadata using NLP techniques.
    SEO is crucial for YouTube success. Using the right keywords can significantly boost your visibility.
    Our AI-powered system provides detailed insights on trending topics and viewer interests.
    """
    
    test_text_ko = """
    유튜브 콘텐츠 제작에 있어서 키워드 분석은 매우 중요합니다. 
    시청자들이 어떤 주제에 관심이 있는지 파악하고, 검색 최적화를 통해 노출을 높일 수 있습니다.
    이 도구는 동영상 스크립트, 댓글, 메타데이터를 분석하여 콘텐츠 제작자에게 유용한 인사이트를 제공합니다.
    자연어 처리 기술을 활용하여 트렌드 키워드와 시청자 관심사를 분석하는 AI 기반 시스템입니다.
    """
    
    # 영어 텍스트 처리 테스트
    print("\n=== 영어 텍스트 키워드 추출 테스트 ===")
    en_extractor = KeywordExtractor(language='en')
    en_results = en_extractor.extract_keywords(test_text_en, top_n=8)
    
    print("\n[영어 키워드 추출 결과]")
    for method, keywords in en_results.items():
        print(f"\n{method.upper()} 키워드:")
        for keyword, score in keywords:
            print(f"  - {keyword} (점수: {score:.4f})")
    
    # 결합된 영어 키워드
    en_combined = en_extractor.combine_keywords(en_results, top_n=10)
    print("\n[영어 통합 키워드 (상위 10개)]")
    for i, kw in enumerate(en_combined, 1):
        print(f"{i}. {kw['keyword']} (점수: {kw['score']:.4f}, 방법: {kw['methods']})")
    
    # 한국어 텍스트 처리 테스트
    print("\n\n=== 한국어 텍스트 키워드 추출 테스트 ===")
    ko_extractor = KeywordExtractor(language='ko')
    ko_results = ko_extractor.extract_keywords(test_text_ko, top_n=8)
    
    print("\n[한국어 키워드 추출 결과]")
    for method, keywords in ko_results.items():
        print(f"\n{method.upper()} 키워드:")
        for keyword, score in keywords:
            print(f"  - {keyword} (점수: {score:.4f})")
    
    # 결합된 한국어 키워드
    ko_combined = ko_extractor.combine_keywords(ko_results, top_n=10)
    print("\n[한국어 통합 키워드 (상위 10개)]")
    for i, kw in enumerate(ko_combined, 1):
        print(f"{i}. {kw['keyword']} (점수: {kw['score']:.4f}, 방법: {kw['methods']})")
    
    # 데이터프레임으로 변환 예시
    df_keywords = ko_extractor.keywords_to_dataframe(ko_results)
    print("\n[한국어 키워드 DataFrame 샘플 (상위 5개)]")
    print(df_keywords.head(5)) 