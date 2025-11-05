import mysql.connector
import os
from dotenv import load_dotenv
import bcrypt

# 加载环境变量
load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.host = "localhost"
        self.database = "video_rag"
        self.user = os.getenv("LOCAL_MYSQL_USER")
        self.password = os.getenv("LOCAL_MYSQL_PWD")
    
    def get_connection(self):
        """获取数据库连接"""
        try:
            connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            return connection
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return None
    
    def init_database(self):
        """初始化数据库表"""
        connection = self.get_connection()
        if not connection:
            return False
            
        try:
            cursor = connection.cursor()
            
            # 创建管理员用户表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    admin_id INT NOT NULL,
                    session_token VARCHAR(255) UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (admin_id) REFERENCES admins(id)
                )
            """)
            
            # 检查是否已存在admin用户
            cursor.execute("SELECT COUNT(*) FROM admins WHERE username = 'admin'")
            count = cursor.fetchone()[0]
            
            # 如果不存在admin用户，则创建默认管理员用户
            if count == 0:
                # 加密默认密码
                default_password = "admin123"
                hashed_password = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt())
                
                cursor.execute("""
                    INSERT INTO admins (username, password) 
                    VALUES (%s, %s)
                """, ("admin", hashed_password))
            
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Exception as e:
            print(f"数据库初始化失败: {e}")
            if connection:
                connection.rollback()
                connection.close()
            return False
    
    def verify_admin(self, username, password):
        """验证管理员用户"""
        connection = self.get_connection()
        if not connection:
            return False
            
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT id, password FROM admins WHERE username = %s",
                (username,)
            )
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if result:
                admin_id, hashed_password = result
                # 确保hashed_password是bytes类型
                if isinstance(hashed_password, str):
                    hashed_password_bytes = hashed_password.encode('utf-8')
                else:
                    hashed_password_bytes = hashed_password
                    
                # 验证密码
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password_bytes):
                    return admin_id
            return False
        except Exception as e:
            print(f"管理员验证失败: {e}")
            if connection:
                connection.close()
            return False
    
    def create_session(self, admin_id, session_token, expires_at):
        """创建会话"""
        connection = self.get_connection()
        if not connection:
            return False
            
        try:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO sessions (admin_id, session_token, expires_at) VALUES (%s, %s, %s)",
                (admin_id, session_token, expires_at)
            )
            connection.commit()
            cursor.close()
            connection.close()
            return True
        except Exception as e:
            print(f"会话创建失败: {e}")
            if connection:
                connection.rollback()
                connection.close()
            return False
    
    def verify_session(self, session_token):
        """验证会话"""
        connection = self.get_connection()
        if not connection:
            return False
            
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT admin_id FROM sessions WHERE session_token = %s AND expires_at > NOW()",
                (session_token,)
            )
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if result:
                return result[0]
            return None
        except Exception as e:
            print(f"会话验证失败: {e}")
            if connection:
                connection.close()
            return None