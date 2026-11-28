"""PDF 기반 RAG 챗봇 화면 (Streamlit).

실행 (이 파일이 있는 폴더에서):
    streamlit run app.py
"""
import os
import tempfile

import streamlit as st

from rag_module import create_rag_chain, page_label

st.set_page_config(page_title="문서 기반 질의응답 챗봇", layout="wide")

st.title("PDF 기반 RAG 시스템")
st.markdown("업로드한 문서에 대해 질문해 보세요.")


def show_sources(sources):
    """답변 아래에 근거 조각을 펼쳐 볼 수 있게 보여 준다."""
    if not sources:
        return
    with st.expander(f"근거 조각 보기 ({len(sources)}개)"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**{i}. {src['page']}쪽**")
            st.text(src["content"])


# 1. 사이드바: 파일 업로드
with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

# 2. 파일 업로드 시 RAG 체인 초기화
if uploaded_file:
    # 다른 PDF를 올리면 체인과 대화 기록을 새로 만든다
    # (원래 예제는 첫 PDF로 만든 체인을 계속 써서, 파일을 바꿔도 이전 문서로 답했음)
    if st.session_state.get("file_name") != uploaded_file.name:
        # 업로드 파일은 프로젝트 폴더가 아닌 OS 임시 폴더에 저장한다
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("문서를 분석 중입니다..."):
            st.session_state.rag_chain = create_rag_chain(temp_path)
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []
        st.success("분석 완료!")

    # 3. 채팅 인터페이스 (메시지 이력 관리)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources"))

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                result = st.session_state.rag_chain.invoke(prompt)
            answer = result["answer"]
            sources = [
                {"page": page_label(d), "content": d.page_content} for d in result["docs"]
            ]
            st.markdown(answer)
            show_sources(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

else:
    st.info("왼쪽 사이드바에서 PDF 파일을 업로드하면 대화가 시작됩니다.")
