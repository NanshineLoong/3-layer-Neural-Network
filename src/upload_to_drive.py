from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import pickle
import json

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    """认证Google Drive API"""
    creds = None
    if os.path.exists('token.json'):
        with open('token.json', 'r') as token:
            creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)
    
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

def upload_file(service, file_path, folder_name="Neural Network Weights"):
    """上传文件到Google Drive的指定文件夹"""
    # 创建文件夹（如果不存在）
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    # 检查文件夹是否已存在
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
        spaces='drive'
    ).execute()
    
    if not results['files']:
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        folder_id = folder.get('id')
    else:
        folder_id = results['files'][0]['id']
    
    # 上传文件
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(
        file_path,
        mimetype='application/octet-stream',
        resumable=True
    )
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    print(f'文件已上传，ID: {file.get("id")}')
    return file.get('id')

def main():
    """主函数"""
    print("开始上传模型权重到Google Drive...")
    
    # 检查权重文件是否存在
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'model_weights.pkl')
    if not os.path.exists(weights_path):
        print("错误：找不到模型权重文件")
        return
    
    try:
        # 认证并创建服务
        creds = authenticate_google_drive()
        service = build('drive', 'v3', credentials=creds)
        
        # 上传文件
        file_id = upload_file(service, weights_path)
        print("上传成功！")
        print(f"您可以在Google Drive中查看文件，文件ID: {file_id}")
        
    except Exception as e:
        print(f"上传过程中出现错误：{str(e)}")

if __name__ == '__main__':
    main() 