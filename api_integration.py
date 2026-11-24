"""
API集成模块
将向量数据库功能集成到Flask应用中
"""

from flask import Blueprint, request, jsonify, send_file
import os
import logging
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from vector_db_manager import VectorDatabaseManager
from vector_retriever import VectorRetriever
from document_loader import DocumentLoader

# 创建蓝图
vector_bp = Blueprint('vector', __name__, url_prefix='/api/vector')

# 全局变量存储向量系统实例
vector_manager: VectorDatabaseManager = None
vector_retriever: VectorRetriever = None

# 临时上传目录
UPLOAD_FOLDER = '/tmp/vector_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def safe_filename(filename: str) -> str:
    """安全化文件名：保留中文等非 ASCII 字符，仅去除路径穿越与危险字符。

    与 werkzeug.utils.secure_filename 不同，后者会把非 ASCII 字符（如中文）
    全部丢弃，导致「报销制度.txt」变成「txt」。这里只做路径穿越防护。
    """
    # 统一路径分隔符后取 basename，防止目录穿越
    filename = os.path.basename(filename.replace("\\", "/"))
    # 去掉控制字符与路径分隔符
    filename = "".join(ch for ch in filename if ch.isprintable() and ch not in ('/', '\\'))
    # 去掉首尾空白与点
    filename = filename.strip().strip(".")
    return filename or "uploaded_file"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_vector_system(
    milvus_host: str = None,
    milvus_port: str = None,
    embedding_model: str = None,
    dashscope_api_key: str = None
):
    """初始化向量系统"""
    global vector_manager, vector_retriever
    
    try:
        vector_manager = VectorDatabaseManager(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            embedding_model=embedding_model,
            dashscope_api_key=dashscope_api_key
        )
        vector_retriever = VectorRetriever(vector_manager)
        logger.info(f"向量系统初始化成功，连接到 Milvus at {milvus_host}:{milvus_port}")
        return True
    except Exception as e:
        logger.error(f"向量系统初始化失败: {str(e)}")
        return False


@vector_bp.route('/upload_document', methods=['POST'])
def upload_document():
    """上传并处理文档"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'file_path' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 file_path 和 collection_name 参数'
            }), 400
        
        file_path = data['file_path']
        collection_name = data['collection_name']
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': f'文件不存在: {file_path}'
            }), 400
        
        # 处理文档
        success = vector_manager.process_file(file_path, collection_name)
        
        if success:
            # 获取数据库信息
            db_info = vector_manager.get_database_info(collection_name)
            
            return jsonify({
                'success': True,
                'message': f'文档处理成功: {file_path}',
                'database_info': db_info
            })
        else:
            return jsonify({
                'success': False,
                'message': f'文档处理失败: {file_path}'
            }), 500
            
    except Exception as e:
        logger.error(f"文档上传API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'文档处理失败: {str(e)}'
        }), 500


@vector_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """上传文件流处理"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '未找到文件部分'}), 400
        
    file = request.files['file']
    collection_name = request.form.get('collection_name', 'agent_rag')
    
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400
        
    if file:
        filename = safe_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            success = vector_manager.process_file(file_path, collection_name)
            if success:
                db_info = vector_manager.get_database_info(collection_name)
                return jsonify({
                    'success': True,
                    'message': f'文件上传并处理成功: {filename}',
                    'database_info': db_info
                })
            else:
                 return jsonify({'success': False, 'message': '文件处理失败'}), 500
        except Exception as e:
             return jsonify({'success': False, 'message': str(e)}), 500
        finally:
            # 可选：处理完后删除临时文件，或者保留
            # os.remove(file_path)
            pass

    return jsonify({'success': False, 'message': '上传失败'}), 500


@vector_bp.route('/query', methods=['POST'])
def query_documents():
    """查询文档"""
    global vector_retriever
    
    if not vector_retriever:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 question 和 collection_name 参数'
            }), 400
        
        question = data['question']
        collection_name = data['collection_name']
        k = data.get('k', 5)  # 返回结果数量
        
        # 执行查询
        result = vector_retriever.answer_question(question, k=k, collection_name=collection_name)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result.answer,
            'confidence': result.confidence,
            'question_type': result.question_type,
            'sources': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in zip(result.source_documents, result.scores)
            ]
        })
        
    except Exception as e:
        logger.error(f"查询API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'查询失败: {str(e)}'
        }), 500


@vector_bp.route('/search', methods=['POST'])
def search_similar():
    """相似性搜索"""
    global vector_retriever
    
    if not vector_retriever:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 query 和 collection_name 参数'
            }), 400
        
        query = data['query']
        collection_name = data['collection_name']
        k = data.get('k', 5)
        
        # 执行相似性搜索
        results = vector_retriever.search_similar_content(query, k=k, collection_name=collection_name)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
        })
        
    except Exception as e:
        logger.error(f"搜索API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'搜索失败: {str(e)}'
        }), 500


@vector_bp.route('/collection_info', methods=['GET'])
def get_collection_info():
    """获取集合信息"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({
            'success': False,
            'message': '请提供 collection_name 参数'
        }), 400

    try:
        db_info = vector_manager.get_database_info(collection_name)
        return jsonify({
            'success': True,
            'database_info': db_info
        })
        
    except Exception as e:
        logger.error(f"数据库信息API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取数据库信息失败: {str(e)}'
        }), 500


@vector_bp.route('/clear_collection', methods=['POST'])
def clear_collection():
    """清空集合"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    data = request.get_json()
    if not data or 'collection_name' not in data:
        return jsonify({
            'success': False,
            'message': '请提供 collection_name 参数'
        }), 400

    collection_name = data['collection_name']

    try:
        vector_manager.clear_database(collection_name)
        return jsonify({
            'success': True,
            'message': f"集合 '{collection_name}' 已清空"
        })

    except Exception as e:
        logger.error(f"清空数据库API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'清空数据库失败: {str(e)}'
        }), 500


@vector_bp.route('/documents', methods=['GET'])
def list_documents():
    """列出已入库文档（按 source 分组）"""
    global vector_manager

    if not vector_manager:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    collection_name = request.args.get('collection_name', 'agent_rag')

    try:
        documents = vector_manager.list_documents(collection_name)
        return jsonify({'success': True, 'documents': documents})
    except Exception as e:
        logger.error(f"列出文档API错误: {str(e)}")
        return jsonify({'success': False, 'message': f'列出文档失败: {str(e)}'}), 500


@vector_bp.route('/delete_document', methods=['POST'])
def delete_document():
    """删除已入库文档（按 source 删除该文档所有分块）"""
    global vector_manager

    if not vector_manager:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    data = request.get_json()
    if not data or 'source' not in data:
        return jsonify({'success': False, 'message': '请提供 source 参数'}), 400

    source = data['source']
    collection_name = data.get('collection_name', 'agent_rag')

    try:
        vector_manager.delete_document(source, collection_name)
        # 同时删除上传目录里的原始文件（若存在）
        original_file = os.path.join(UPLOAD_FOLDER, os.path.basename(source))
        if os.path.exists(original_file):
            os.remove(original_file)
        return jsonify({'success': True, 'message': f'文档 {source} 已删除'})
    except Exception as e:
        logger.error(f"删除文档API错误: {str(e)}")
        return jsonify({'success': False, 'message': f'删除文档失败: {str(e)}'}), 500


@vector_bp.route('/download_document', methods=['GET'])
def download_document():
    """下载已入库文档（优先原始文件，否则从 Milvus 拼接文本）"""
    global vector_manager

    if not vector_manager:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    source = request.args.get('source')
    collection_name = request.args.get('collection_name', 'agent_rag')

    if not source:
        return jsonify({'success': False, 'message': '请提供 source 参数'}), 400

    try:
        # 优先返回上传目录里的原始文件
        original_file = os.path.join(UPLOAD_FOLDER, os.path.basename(source))
        if os.path.exists(original_file):
            return send_file(original_file, as_attachment=True, download_name=source)

        # 否则从 Milvus 取全部 chunk 按顺序拼回文本下载
        from io import BytesIO
        text = vector_manager.get_document_text(source, collection_name)
        buf = BytesIO(text.encode('utf-8'))
        return send_file(buf, as_attachment=True, download_name=source,
                         mimetype='text/plain; charset=utf-8')
    except Exception as e:
        logger.error(f"下载文档API错误: {str(e)}")
        return jsonify({'success': False, 'message': f'下载文档失败: {str(e)}'}), 500


# 错误处理
@vector_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': '接口不存在'
    }), 404


@vector_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': '服务器内部错误'
    }), 500


def register_vector_routes(app):
    """注册向量数据库路由到Flask应用"""
    app.register_blueprint(vector_bp)
    
    # 自动初始化向量系统
    with app.app_context():
        init_vector_system()
    
    logger.info("向量数据库API路由已注册")