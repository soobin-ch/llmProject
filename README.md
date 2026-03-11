# 📋 외국인 체류지변경 신고 보조 서비스 (Foreigner Residence Change Assistant)

이 프로젝트는 한국에 거주하는 외국인이 이사 후 필수적으로 진행해야 하는 **체류지 변경 신고** 절차를 돕기 위한 Streamlit 기반 웹 애플리케이션입니다. Upstage의 **Document Parse** 기술과 **Solar LLM**을 활용하여 복잡한 법정 서식을 분석하고 다국어 안내 및 질의응답을 제공합니다.

## 🌟 주요 기능 (Key Features)

### 1. 지능형 질의응답 (RAG 기반)
* **문서 기반 가이드**: "외국인 체류지변경 신고서" PDF 내용을 바탕으로 신고 기한(15일 이내) 및 필요 서류를 정확히 안내합니다
* **Upstage Layout Analysis**: PDF 내의 복잡한 표 구조를 분석하여 신청인 정보, 동반자 정보 등을 정확하게 추출합니다.

### 2. 다국어 지원 (Multilingual Support)
* **English Interface**: 영어권 사용자를 위한 절차 안내, 서식 링크 제공 및 샘플 PDF 확인 기능을 지원합니다.
* **중국어 인터페이스 (中文)**: 중국어 사용자를 위해 Upstage Solar 모델을 활용한 전용 상담 페이지와 자동 번역 질의응답을 제공합니다.

### 3. 디지털 서식 작성 보조
* **입력 항목**: 성명(영문/한자), 생년월일, 국적, 외국인등록번호 등 서식에 필요한 필수 정보를 웹상에서 미리 입력해 볼 수 있습니다.
* **주소 및 연락처**: 이전 체류지, 새로운 체류지 주소 및 전화번호 입력 기능을 제공합니다.
* **동반자(Dependents) 관리**: 한국 내 동반 가족의 성명, 생년월일, 관계, 등록번호 항목을 포함합니다.

## 🛠 기술 스택 (Tech Stack)

| 구분 | 기술 |
| :--- | :--- |
| **Frontend** | Streamlit |
| **LLM** | Upstage ChatUpstage (Solar) |
| **Document AI** | Upstage Document Parse Loader |
| **Vector DB** | FAISS |
| **Orchestration** | LangChain |

## 📖 주요 신고 항목 (Main Report Fields)
제공된 PDF 서식을 기준으로 다음 데이터가 처리됩니다:
* **신청인 정보**: Surname, Given names, 한자 성명, 성별, 생년월일, 국적.
* **체류 정보**: 외국인등록번호, 등록일자, 전 체류지, 신 체류지.
* **동반자 정보**: 성명, 생년월일, 성별, 관계, 등록번호.
* **인증**: 신고일자 및 신고인 서명란.

## 🚀 시작하기 (Getting Started)

### 1. 환경 변수 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 Upstage API 키를 입력하세요.
```env
UPSTAGE_API_KEY=your_api_key_here
