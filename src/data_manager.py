"""
데이터 저장 및 결과 리포팅 모듈 - 수집된 데이터 통합, 저장 및 분석 결과 생성
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.nlp_processor import KeywordExtractor


class DataManager:
    """수집 데이터 통합 및 저장 관리 클래스"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        DataManager 초기화
        
        Args:
            base_dir (str, optional): 데이터 저장 기본 디렉토리
                                    미지정시 프로젝트 루트의 'data' 폴더 사용
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # 프로젝트 루트 디렉토리 경로 찾기
            project_root = Path(__file__).parent.parent.resolve()
            self.base_dir = project_root / "data"
        
        # 필요한 디렉토리 생성
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 구조 생성"""
        directories = [
            self.base_dir,
            self.base_dir / "keywords",
            self.base_dir / "videos",
            self.base_dir / "channels",
            self.base_dir / "transcripts",
            self.base_dir / "comments",
            self.base_dir / "analysis",
            self.base_dir / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"데이터 저장 디렉토리 준비 완료: {self.base_dir}")
    
    def save_df_to_csv(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        subdir: Optional[str] = None
    ) -> str:
        """
        DataFrame을 CSV 파일로 저장
        
        Args:
            df (pd.DataFrame): 저장할 DataFrame
            filename (str): 파일명 (확장자 없이)
            subdir (str, optional): 하위 디렉토리 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        if df.empty:
            print(f"경고: 저장할 데이터가 없습니다 ({filename}).")
            return ""
        
        # 경로 설정
        if subdir:
            save_dir = self.base_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.base_dir
        
        # 확장자 확인 및 추가
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # 전체 파일 경로
        file_path = save_dir / filename
        
        # 저장
        try:
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"파일 저장 완료: {file_path}")
            return str(file_path)
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return ""
    
    def save_dict_to_json(
        self, 
        data: Dict, 
        filename: str, 
        subdir: Optional[str] = None
    ) -> str:
        """
        Dictionary를 JSON 파일로 저장
        
        Args:
            data (Dict): 저장할 데이터
            filename (str): 파일명 (확장자 없이)
            subdir (str, optional): 하위 디렉토리 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        if not data:
            print(f"경고: 저장할 데이터가 없습니다 ({filename}).")
            return ""
        
        # 경로 설정
        if subdir:
            save_dir = self.base_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.base_dir
        
        # 확장자 확인 및 추가
        if not filename.endswith('.json'):
            filename += '.json'
        
        # 전체 파일 경로
        file_path = save_dir / filename
        
        # 저장
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"파일 저장 완료: {file_path}")
            return str(file_path)
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return ""
    
    def load_df_from_csv(
        self, 
        filename: str, 
        subdir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        CSV 파일에서 DataFrame 로드
        
        Args:
            filename (str): 파일명
            subdir (str, optional): 하위 디렉토리 이름
            
        Returns:
            pd.DataFrame: 로드된 DataFrame
        """
        # 경로 설정
        if subdir:
            file_dir = self.base_dir / subdir
        else:
            file_dir = self.base_dir
        
        # 확장자 확인 및 추가
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # 전체 파일 경로
        file_path = file_dir / filename
        
        # 로드
        try:
            if not file_path.exists():
                print(f"파일을 찾을 수 없습니다: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(file_path)
            print(f"파일 로드 완료: {file_path} (행 수: {len(df)})")
            return df
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def load_dict_from_json(
        self, 
        filename: str, 
        subdir: Optional[str] = None
    ) -> Dict:
        """
        JSON 파일에서 Dictionary 로드
        
        Args:
            filename (str): 파일명
            subdir (str, optional): 하위 디렉토리 이름
            
        Returns:
            Dict: 로드된 Dictionary
        """
        # 경로 설정
        if subdir:
            file_dir = self.base_dir / subdir
        else:
            file_dir = self.base_dir
        
        # 확장자 확인 및 추가
        if not filename.endswith('.json'):
            filename += '.json'
        
        # 전체 파일 경로
        file_path = file_dir / filename
        
        # 로드
        try:
            if not file_path.exists():
                print(f"파일을 찾을 수 없습니다: {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"파일 로드 완료: {file_path}")
            return data
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            return {}


class ReportGenerator:
    """데이터 분석 및 리포트 생성 클래스"""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        ReportGenerator 초기화
        
        Args:
            data_manager (DataManager, optional): 데이터 관리자 인스턴스
                                                미지정시 새로 생성
        """
        self.data_manager = data_manager or DataManager()
        self.keyword_extractor = None  # 필요시 초기화
    
    def _init_keyword_extractor(self, language='ko'):
        """필요시 키워드 추출기 초기화"""
        if self.keyword_extractor is None:
            from src.nlp_processor import KeywordExtractor
            self.keyword_extractor = KeywordExtractor(language=language)
    
    def analyze_comments(
        self,
        video_id: str,
        comments_df: Optional[pd.DataFrame] = None,
        top_n: int = 20,
        language: str = 'ko'
    ) -> Dict[str, Any]:
        """
        비디오 댓글 분석 수행
        
        Args:
            video_id (str): 동영상 ID
            comments_df (pd.DataFrame, optional): 댓글 DataFrame
                                            미지정시 저장된 파일에서 로드
            top_n (int): 추출할 상위 키워드 수
            language (str): 텍스트 언어
            
        Returns:
            Dict: 분석 결과
        """
        self._init_keyword_extractor(language)
        
        # 댓글 데이터 확보
        if comments_df is None or comments_df.empty:
            comments_df = self.data_manager.load_df_from_csv(
                f"{video_id}_comments", 
                subdir="comments"
            )
        
        if comments_df.empty:
            print(f"댓글 분석할 데이터가 없습니다 (video_id: {video_id}).")
            return {}
        
        # 댓글 텍스트 추출
        all_comments_text = " ".join(comments_df['text'].dropna().astype(str).tolist())
        
        if not all_comments_text.strip():
            print("분석할 댓글 텍스트가 없습니다.")
            return {}
        
        # 키워드 분석
        print(f"댓글 텍스트 키워드 분석 시작 (글자 수: {len(all_comments_text)})...")
        
        # 키워드 추출
        keywords_results = self.keyword_extractor.extract_keywords(
            all_comments_text,
            methods=['yake', 'rake'],  # KeyBERT는 상대적으로 느려서 선택적 사용
            top_n=top_n,
            preprocess=True
        )
        
        # 통합 키워드
        combined_keywords = self.keyword_extractor.combine_keywords(
            keywords_results,
            top_n=top_n
        )
        
        # 결과 통합
        result = {
            'video_id': video_id,
            'comment_count': len(comments_df),
            'total_chars': len(all_comments_text),
            'keywords': keywords_results,
            'combined_keywords': combined_keywords
        }
        
        # 결과 저장
        self.data_manager.save_dict_to_json(
            result,
            f"{video_id}_comment_analysis",
            subdir="analysis"
        )
        
        print(f"댓글 분석 완료 (video_id: {video_id}, 키워드 수: {len(combined_keywords)})")
        return result
    
    def analyze_transcript(
        self,
        video_id: str,
        transcript_df: Optional[pd.DataFrame] = None,
        top_n: int = 20,
        language: str = 'ko'
    ) -> Dict[str, Any]:
        """
        비디오 스크립트 분석 수행
        
        Args:
            video_id (str): 동영상 ID
            transcript_df (pd.DataFrame, optional): 스크립트 DataFrame
                                              미지정시 저장된 파일에서 로드
            top_n (int): 추출할 상위 키워드 수
            language (str): 텍스트 언어
            
        Returns:
            Dict: 분석 결과
        """
        self._init_keyword_extractor(language)
        
        # 스크립트 데이터 확보
        if transcript_df is None or transcript_df.empty:
            transcript_df = self.data_manager.load_df_from_csv(
                f"{video_id}_transcript", 
                subdir="transcripts"
            )
        
        if transcript_df.empty:
            print(f"스크립트 분석할 데이터가 없습니다 (video_id: {video_id}).")
            return {}
        
        # 스크립트 텍스트 추출 (cleaned_transcript 열 사용)
        transcript_text = ""
        if 'cleaned_transcript' in transcript_df.columns:
            transcript_text = transcript_df['cleaned_transcript'].iloc[0]
        elif 'raw_transcript' in transcript_df.columns:
            transcript_text = transcript_df['raw_transcript'].iloc[0]
        
        if not transcript_text or not isinstance(transcript_text, str):
            print("분석할 스크립트 텍스트가 없습니다.")
            return {}
        
        # 키워드 분석
        print(f"스크립트 텍스트 키워드 분석 시작 (글자 수: {len(transcript_text)})...")
        
        # 키워드 추출
        keywords_results = self.keyword_extractor.extract_keywords(
            transcript_text,
            methods=['yake', 'rake', 'keybert'],  # 스크립트는 상대적으로 짧으므로 KeyBERT 포함
            top_n=top_n,
            preprocess=True
        )
        
        # 통합 키워드
        combined_keywords = self.keyword_extractor.combine_keywords(
            keywords_results,
            top_n=top_n
        )
        
        # 결과 통합
        result = {
            'video_id': video_id,
            'transcript_length': len(transcript_text),
            'keywords': keywords_results,
            'combined_keywords': combined_keywords
        }
        
        # 결과 저장
        self.data_manager.save_dict_to_json(
            result,
            f"{video_id}_transcript_analysis",
            subdir="analysis"
        )
        
        print(f"스크립트 분석 완료 (video_id: {video_id}, 키워드 수: {len(combined_keywords)})")
        return result
    
    def generate_wordcloud(
        self,
        keywords: List[Dict[str, Any]],
        title: str,
        output_filename: str,
        subdir: str = "reports",
        width: int = 800,
        height: int = 400
    ) -> str:
        """
        키워드 워드클라우드 생성 및 저장
        
        Args:
            keywords (List[Dict]): combine_keywords() 결과
            title (str): 워드클라우드 제목
            output_filename (str): 출력 파일명 (확장자 없이)
            subdir (str): 저장 하위 디렉토리
            width (int): 이미지 너비
            height (int): 이미지 높이
            
        Returns:
            str: 저장된 이미지 파일 경로
        """
        if not keywords:
            print("워드클라우드 생성을 위한 키워드가 없습니다.")
            return ""
        
        try:
            # 워드클라우드 데이터 준비 (키워드와 가중치)
            word_weights = {k['keyword']: k['score'] * 100 for k in keywords}
            
            # 워드클라우드 생성
            wc = WordCloud(
                font_path="malgun",  # 한글 폰트 (없을 경우 시스템 기본 폰트 사용)
                width=width, 
                height=height,
                background_color='white'
            )
            
            # 키워드-가중치 데이터로 워드클라우드 생성
            wc.generate_from_frequencies(word_weights)
            
            # 그림 생성
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(title)
            plt.tight_layout(pad=0)
            
            # 파일 경로 설정
            if not output_filename.endswith('.png'):
                output_filename += '.png'
            
            # 저장 디렉토리 설정
            save_dir = self.data_manager.base_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 전체 파일 경로
            file_path = save_dir / output_filename
            
            # 저장
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            
            print(f"워드클라우드 이미지 저장 완료: {file_path}")
            return str(file_path)
        
        except Exception as e:
            print(f"워드클라우드 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def create_comprehensive_report(
        self,
        video_id: str,
        include_wordcloud: bool = True
    ) -> Dict[str, Any]:
        """
        동영상에 대한 종합 리포트 생성
        
        Args:
            video_id (str): 동영상 ID
            include_wordcloud (bool): 워드클라우드 포함 여부
            
        Returns:
            Dict: 종합 리포트 데이터
        """
        # 필요한 데이터 로드
        video_info = self.data_manager.load_dict_from_json(
            f"{video_id}_info", 
            subdir="videos"
        )
        
        transcript_analysis = self.data_manager.load_dict_from_json(
            f"{video_id}_transcript_analysis", 
            subdir="analysis"
        )
        
        comment_analysis = self.data_manager.load_dict_from_json(
            f"{video_id}_comment_analysis", 
            subdir="analysis"
        )
        
        # 데이터가 부족한 경우 분석 수행
        if not transcript_analysis:
            print(f"스크립트 분석 결과가 없어 분석을 수행합니다.")
            transcript_analysis = self.analyze_transcript(video_id)
        
        if not comment_analysis:
            print(f"댓글 분석 결과가 없어 분석을 수행합니다.")
            comment_analysis = self.analyze_comments(video_id)
        
        # 종합 리포트 생성
        report = {
            'video_id': video_id,
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'video_info': video_info,
            'transcript_keywords': transcript_analysis.get('combined_keywords', []),
            'comment_keywords': comment_analysis.get('combined_keywords', [])
        }
        
        # 워드클라우드 생성 (선택 사항)
        if include_wordcloud:
            # 스크립트 워드클라우드
            if transcript_analysis and 'combined_keywords' in transcript_analysis:
                script_wc_path = self.generate_wordcloud(
                    transcript_analysis['combined_keywords'],
                    f"Video Script Keywords ({video_id})",
                    f"{video_id}_script_wordcloud",
                    subdir="reports"
                )
                report['script_wordcloud_path'] = script_wc_path
            
            # 댓글 워드클라우드
            if comment_analysis and 'combined_keywords' in comment_analysis:
                comment_wc_path = self.generate_wordcloud(
                    comment_analysis['combined_keywords'],
                    f"Video Comments Keywords ({video_id})",
                    f"{video_id}_comments_wordcloud",
                    subdir="reports"
                )
                report['comments_wordcloud_path'] = comment_wc_path
        
        # 종합 리포트 저장
        self.data_manager.save_dict_to_json(
            report,
            f"{video_id}_comprehensive_report",
            subdir="reports"
        )
        
        print(f"동영상 '{video_id}'에 대한 종합 리포트 생성 완료")
        return report
    
    def generate_batch_analysis(
        self,
        video_ids: List[str],
        include_wordclouds: bool = True
    ) -> Dict[str, Any]:
        """
        여러 동영상에 대한 일괄 분석 수행
        
        Args:
            video_ids (List[str]): 분석할 동영상 ID 목록
            include_wordclouds (bool): 워드클라우드 포함 여부
            
        Returns:
            Dict: 일괄 분석 결과 요약
        """
        start_time = time.time()
        print(f"총 {len(video_ids)}개 동영상 일괄 분석 시작...")
        
        results = {
            'total_videos': len(video_ids),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'video_reports': []
        }
        
        for i, video_id in enumerate(video_ids):
            print(f"\n[{i+1}/{len(video_ids)}] 동영상 '{video_id}' 분석 중...")
            
            try:
                # 종합 리포트 생성
                report = self.create_comprehensive_report(
                    video_id=video_id,
                    include_wordcloud=include_wordclouds
                )
                
                if report:
                    results['successful_analyses'] += 1
                    results['video_reports'].append({
                        'video_id': video_id,
                        'status': 'success',
                        'report_file': f"{video_id}_comprehensive_report.json"
                    })
                else:
                    results['failed_analyses'] += 1
                    results['video_reports'].append({
                        'video_id': video_id,
                        'status': 'failed',
                        'reason': '리포트 생성 실패'
                    })
                    
            except Exception as e:
                print(f"동영상 '{video_id}' 분석 중 오류 발생: {e}")
                results['failed_analyses'] += 1
                results['video_reports'].append({
                    'video_id': video_id,
                    'status': 'failed',
                    'reason': str(e)
                })
        
        # 경과 시간 계산
        elapsed_time = time.time() - start_time
        results['elapsed_time'] = elapsed_time
        
        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.data_manager.save_dict_to_json(
            results,
            f"batch_analysis_summary_{timestamp}",
            subdir="reports"
        )
        
        print(f"\n일괄 분석 완료:")
        print(f"  - 총 동영상 수: {len(video_ids)}")
        print(f"  - 성공: {results['successful_analyses']}")
        print(f"  - 실패: {results['failed_analyses']}")
        print(f"  - 소요 시간: {elapsed_time:.1f}초")
        
        return results


# 테스트 코드
if __name__ == "__main__":
    # 테스트용 데이터 매니저
    print("데이터 저장 및 리포팅 모듈 테스트")
    data_manager = DataManager()
    
    # 기본 디렉토리 구조 생성 테스트
    print(f"기본 데이터 디렉토리: {data_manager.base_dir}")
    
    # 샘플 데이터 생성 및 저장 테스트
    sample_df = pd.DataFrame({
        'keyword': ['파이썬', '유튜브', '분석', '코딩', 'API'],
        'score': [1.0, 0.8, 0.7, 0.6, 0.5]
    })
    
    data_manager.save_df_to_csv(
        sample_df,
        "sample_keywords",
        subdir="keywords"
    )
    
    # 리포트 생성기 테스트
    report_generator = ReportGenerator(data_manager)
    
    print("\n리포트 생성기 초기화 완료")
    print("실제 데이터로 테스트하려면 'video_id'를 지정하여 analyze_comments() 또는 analyze_transcript() 메서드 사용") 