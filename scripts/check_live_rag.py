"""显式运行的真实模型 + Milvus 检查，不属于普通单元测试。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def make_pdf(text):
    """构造单页文字 PDF，供真实上传检查使用。"""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=200)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 30 120 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description='真实 RAG 联调，会调用 DashScope 并产生用量')
    parser.add_argument('--run', action='store_true', help='确认执行真实模型调用')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return

    from dotenv import load_dotenv
    load_dotenv(PROJECT / '.env')
    if not os.getenv('DASHSCOPE_API_KEY') or not os.getenv('EMBEDDING_MODEL'):
        raise RuntimeError('请配置本地模型密钥及向量模型')
    # 独立集合前缀 + 临时目录，业务上传文件、缓存和集合均不使用。
    os.environ['VECTOR_BACKEND'] = 'milvus'
    os.environ['MILVUS_COLLECTION'] = 'ask_vault_live_test'
    from knowledge_base import KnowledgeBase
    import knowledge_base
    from server import create_app
    from pymilvus import MilvusClient

    with tempfile.TemporaryDirectory(prefix='askvault-live-') as temporary:
        root = Path(temporary)
        folder = root / 'data/uploads'
        app = create_app(KnowledgeBase(root))
        app.config.update(TESTING=True, UPLOAD_FOLDER=folder)
        client = app.test_client()
        document_ids = []
        prior_indexes = []
        try:
            for token in ['蓝鲸七号', '白鹭九号']:
                response = client.post('/api/upload', data={
                    'file': (io.BytesIO(f'验证仓库的通行口令是{token}。'.encode()), '联调资料.txt'),
                })
                assert response.status_code == 201, '上传失败'
                document_ids.append(response.json['document']['id'])
                response.close()
            docs = client.get('/api/documents').json['documents']
            assert len(docs) == 2 and docs[0]['filename'] == docs[1]['filename']
            print('PASS: two same-name documents have distinct IDs', flush=True)

            with patch('knowledge_base.embed_text', wraps=knowledge_base.embed_text) as observed:
                for position, (document_id, expected) in enumerate([
                    (document_ids[0], '蓝鲸七号'), (document_ids[1], '白鹭九号'),
                    (document_ids[0], '蓝鲸七号'),
                ]):
                    before = len([call for call in observed.call_args_list if call.kwargs.get('text_type', 'document') == 'document'])
                    response = client.post('/api/query', json={
                        'question': '验证仓库的通行口令是什么？', 'document_id': document_id, 'top_k': 2,
                    })
                    result = response.json
                    assert response.status_code == 200 and result['success'], '真实问答接口失败'
                    assert result['backend'] == 'milvus'
                    assert expected in result['answer'], '回答没有包含资料中的口令'
                    wrong = '白鹭九号' if expected == '蓝鲸七号' else '蓝鲸七号'
                    assert wrong not in result['answer'], '回答混入另一份资料'
                    assert result['sources'] and '[1]' in result['answer'], '缺少引用'
                    assert all(item['source'] == '联调资料.txt' for item in result['sources'])
                    assert all(wrong not in match['text'] for match in result['matches'])
                    after = len([call for call in observed.call_args_list if call.kwargs.get('text_type', 'document') == 'document'])
                    assert after - before == (0 if position == 2 else 1), '文档缓存复用与预期不符'
                    print(f"PASS query {position + 1}: {result['answer']} score={result['matches'][0]['score']:.4f}; document embedding calls={after - before}", flush=True)
                    response.close()

            # 同一文档版本的入库数量保持一条，不因重复提问增加。
            for document_id in document_ids:
                cached = json.loads((folder / f'{document_id}.index.json').read_text(encoding='utf-8'))
                from milvus_store import MilvusVectorStore
                store = MilvusVectorStore(
                    os.getenv('MILVUS_URI', 'http://127.0.0.1:19630'), cached['metadata'],
                    len(cached['records'][0]['embedding']), 'ask_vault_live_test',
                )
                try:
                    assert store.sync_records(cached['records']) is False, '重复入库未被识别'
                    rows = store.client.query(collection_name=store.collection_name,
                        filter=f'index_key == "{store.index_key}"', output_fields=['count(*)'])
                    assert rows[0]['count(*)'] == 1, '同一文档版本出现重复片段'
                finally:
                    store.close()
            print('PASS: repeated questions do not duplicate Milvus rows', flush=True)

            with client.post('/api/query', json={
                'question': '分别列出两份验证仓库资料中的通行口令。',
                'document_ids': document_ids, 'top_k': 2,
            }) as joint:
                assert joint.status_code == 200, '联合检索请求失败'
                assert {item['document_id'] for item in joint.json['matches']} == set(document_ids)
                assert {item['document_id'] for item in joint.json['sources']} == set(document_ids)
                assert all(word in joint.json['answer'] for word in ('蓝鲸七号', '白鹭九号'))
            print('PASS: joint retrieval returns both same-name documents and references', flush=True)

            from docx import Document
            word = Document()
            word.add_paragraph('The archive access code is listed in the following table.')
            cells = word.add_table(rows=1, cols=2).rows[0].cells
            cells[0].text, cells[1].text = 'Archive access code', 'Cobalt Nine'
            buffer = io.BytesIO()
            word.save(buffer)
            formats = [
                ('regional.pdf', make_pdf('The regional archive access code is Cedar Seven.'), 'Cedar Seven'),
                ('table.docx', buffer.getvalue(), 'Cobalt Nine'),
            ]
            for filename, content, expected in formats:
                with client.post('/api/upload', data={'file': (io.BytesIO(content), filename)}) as uploaded:
                    assert uploaded.status_code == 201, f'{filename} 上传失败'
                    document_id = uploaded.json['document']['id']
                document_ids.append(document_id)
                with client.get(f'/api/documents/{document_id}/download') as downloaded:
                    assert downloaded.status_code == 200
                    assert downloaded.data == content and filename in downloaded.headers['Content-Disposition']
                with client.post('/api/query', json={
                    'question': 'What is the archive access code?', 'document_id': document_id,
                }) as queried:
                    assert queried.status_code == 200, f'{filename} 问答失败'
                    assert expected.lower() in queried.json['answer'].lower(), f'{filename} 答案未找到原文'
                    assert queried.json['sources'][0]['source'] == filename
                print(f'PASS: {filename} upload, download and real answer', flush=True)

            document_id = document_ids[0]
            cache_path = folder / f'{document_id}.index.json'
            prior_indexes.append(json.loads(cache_path.read_text(encoding='utf-8')))
            payload = '验证仓库的通行口令是金鹿八号。'.encode()
            with client.put(f'/api/documents/{document_id}', data={
                'file': (io.BytesIO(payload), 'replacement.txt'),
            }) as replaced:
                assert replaced.status_code == 200
                assert replaced.json['document']['id'] == document_id
            with client.post('/api/query', json={
                'question': '验证仓库的通行口令是什么？', 'document_id': document_id,
            }) as updated:
                assert updated.status_code == 200
                assert '金鹿八号' in updated.json['answer'] and '蓝鲸七号' not in updated.json['answer']
                assert all('蓝鲸七号' not in item['text'] for item in updated.json['matches'])
            with client.delete(f'/api/documents/{document_id}') as removed:
                assert removed.status_code == 200
            with client.get('/api/documents') as listed:
                assert {item['id'] for item in listed.json['documents']} == set(document_ids[1:])
            with client.get(f'/api/documents/{document_id}/download') as missing:
                assert missing.status_code == 404
            with client.post('/api/query', json={'question': '通行口令是什么？', 'document_id': document_id}) as missing:
                assert missing.status_code == 404
            print('PASS: replace invalidates stale retrieval; removed file is excluded and recoverable', flush=True)
            assert (folder / f'{document_id}.txt').exists()
            assert (folder / f'{document_id}.deleted.json').exists()

        finally:
            # 只删除本次 UUID 文档对应的测试行；不删除集合或其他测试记录。
            database = MilvusClient(uri=os.getenv('MILVUS_URI', 'http://127.0.0.1:19630'), timeout=15)
            try:
                current_indexes = [json.loads(path.read_text(encoding='utf-8'))
                                   for path in folder.glob('*.index.json')]
                for cached in prior_indexes + current_indexes:
                    metadata = cached['metadata']
                    assert metadata['document_id'] in document_ids
                    model_key = hashlib.sha256(metadata['embedding_model'].encode()).hexdigest()[:12]
                    collection = f"ask_vault_live_test_{model_key}_{len(cached['records'][0]['embedding'])}"
                    index_key = hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    if database.has_collection(collection_name=collection):
                        database.delete(collection_name=collection, filter=f'index_key == "{index_key}"')
                        database.flush(collection_name=collection)
                        rows = database.query(collection_name=collection, filter=f'index_key == "{index_key}"', output_fields=['count(*)'], consistency_level='Strong')
                        assert rows[0]['count(*)'] == 0, '测试向量未清理完成'
            finally:
                database.close()
            print('Cleaned this run\'s temporary documents, caches and Milvus rows', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # 不把云服务原始异常（可能带请求信息）直接输出。
        detail = str(error) if isinstance(error, AssertionError) else type(error).__name__
        print(f'Live check failed: {detail}', file=sys.stderr)
        sys.exit(1)
