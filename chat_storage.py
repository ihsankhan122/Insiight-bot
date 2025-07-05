import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SQLiteChatStorage:
    """SQLite-based chat storage for InsightBot"""
    
    def __init__(self, db_path="insightbot_chats.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        try:
            with self.get_db() as conn:
                # Users table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Chats table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        title TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,  -- JSON string
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Messages table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER,
                        message_type TEXT,  -- 'user' or 'assistant'
                        content TEXT,
                        content_segments TEXT,  -- JSON for complex content
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (chat_id) REFERENCES chats (id)
                    )
                """)
                
                # Chat files table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER,
                        filename TEXT,
                        file_size INTEGER,
                        file_type TEXT,
                        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (chat_id) REFERENCES chats (id)
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
                
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def get_db(self):
        """Database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def ensure_user_exists(self, username: str) -> int:
        """Ensure user exists and return user ID"""
        with self.get_db() as conn:
            # Try to get existing user
            cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                return row[0]
            else:
                # Create new user
                cursor = conn.execute(
                    "INSERT INTO users (username) VALUES (?)",
                    (username,)
                )
                return cursor.lastrowid
    
    def create_chat(self, username: str, title: str = None, metadata: dict = None) -> int:
        """Create new chat session"""
        user_id = self.ensure_user_exists(username)
        
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        with self.get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO chats (user_id, title, metadata)
                VALUES (?, ?, ?)
            """, (user_id, title, json.dumps(metadata or {})))
            
            chat_id = cursor.lastrowid
            logger.info(f"Created new chat {chat_id} for user {username}")
            return chat_id
    
    def save_message(self, chat_id: int, message_type: str, content: str, content_segments: list = None):
        """Save a single message"""
        try:
            with self.get_db() as conn:
                conn.execute("""
                    INSERT INTO messages (chat_id, message_type, content, content_segments)
                    VALUES (?, ?, ?, ?)
                """, (chat_id, message_type, content, json.dumps(content_segments)))
                
                # Update chat's updated_at timestamp
                conn.execute("""
                    UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
                """, (chat_id,))
                
            logger.debug(f"Saved {message_type} message to chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            raise
    
    def save_chat_files(self, chat_id: int, files_info: List[Dict]):
        """Save file information for a chat"""
        try:
            with self.get_db() as conn:
                for file_info in files_info:
                    conn.execute("""
                        INSERT INTO chat_files (chat_id, filename, file_size, file_type)
                        VALUES (?, ?, ?, ?)
                    """, (
                        chat_id,
                        file_info['name'],
                        file_info['size'],
                        file_info['name'].split('.')[-1] if '.' in file_info['name'] else 'unknown'
                    ))
            logger.info(f"Saved {len(files_info)} file records for chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save chat files: {e}")
            raise
    
    def get_user_chats(self, username: str, limit: int = 50) -> List[Dict]:
        """Get all chats for a user"""
        try:
            user_id = self.ensure_user_exists(username)
            
            with self.get_db() as conn:
                rows = conn.execute("""
                    SELECT 
                        c.id, 
                        c.title, 
                        c.created_at, 
                        c.updated_at, 
                        c.metadata,
                        (SELECT COUNT(*) FROM messages WHERE chat_id = c.id) as message_count,
                        (SELECT COUNT(*) FROM chat_files WHERE chat_id = c.id) as file_count
                    FROM chats c
                    WHERE c.user_id = ? AND c.is_active = 1
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                """, (user_id, limit)).fetchall()
                
                chats = []
                for row in rows:
                    chat = {
                        'id': row['id'],
                        'title': row['title'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                        'message_count': row['message_count'],
                        'file_count': row['file_count']
                    }
                    chats.append(chat)
                
                return chats
        except Exception as e:
            logger.error(f"Failed to get user chats: {e}")
            return []
    
    def load_chat_messages(self, chat_id: int) -> List[Dict]:
        """Load all messages for a chat"""
        try:
            with self.get_db() as conn:
                rows = conn.execute("""
                    SELECT message_type, content, content_segments, timestamp
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY timestamp ASC
                """, (chat_id,)).fetchall()
                
                messages = []
                for row in rows:
                    message = {
                        "type": row["message_type"],
                        "content": row["content"],
                        "timestamp": row["timestamp"]
                    }
                    
                    # Parse content_segments if available
                    if row["content_segments"]:
                        try:
                            content_segments = json.loads(row["content_segments"])
                            message["content_segments"] = content_segments
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse content_segments for message in chat {chat_id}")
                    
                    messages.append(message)
                
                return messages
        except Exception as e:
            logger.error(f"Failed to load chat messages: {e}")
            return []
    
    def load_chat_files(self, chat_id: int) -> List[Dict]:
        """Load file information for a chat"""
        try:
            with self.get_db() as conn:
                rows = conn.execute("""
                    SELECT filename, file_size, file_type, upload_timestamp
                    FROM chat_files
                    WHERE chat_id = ?
                    ORDER BY upload_timestamp ASC
                """, (chat_id,)).fetchall()
                
                files = []
                for row in rows:
                    file_info = {
                        'name': row['filename'],
                        'size': row['file_size'],
                        'type': row['file_type'],
                        'uploaded_at': row['upload_timestamp']
                    }
                    files.append(file_info)
                
                return files
        except Exception as e:
            logger.error(f"Failed to load chat files: {e}")
            return []
    
    def update_chat_title(self, chat_id: int, title: str):
        """Update chat title"""
        try:
            with self.get_db() as conn:
                conn.execute("""
                    UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
                """, (title, chat_id))
            logger.info(f"Updated title for chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to update chat title: {e}")
            raise
    
    def delete_chat(self, chat_id: int, username: str):
        """Soft delete a chat (mark as inactive)"""
        try:
            user_id = self.ensure_user_exists(username)
            
            with self.get_db() as conn:
                # Verify chat belongs to user before deleting
                cursor = conn.execute("""
                    SELECT id FROM chats WHERE id = ? AND user_id = ?
                """, (chat_id, user_id))
                
                if cursor.fetchone():
                    conn.execute("""
                        UPDATE chats SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?
                    """, (chat_id,))
                    logger.info(f"Deleted chat {chat_id} for user {username}")
                    return True
                else:
                    logger.warning(f"Chat {chat_id} not found or doesn't belong to user {username}")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete chat: {e}")
            return False
    
    def get_chat_info(self, chat_id: int) -> Optional[Dict]:
        """Get basic chat information"""
        try:
            with self.get_db() as conn:
                row = conn.execute("""
                    SELECT c.title, c.created_at, c.metadata, u.username
                    FROM chats c
                    JOIN users u ON c.user_id = u.id
                    WHERE c.id = ? AND c.is_active = 1
                """, (chat_id,)).fetchone()
                
                if row:
                    return {
                        'id': chat_id,
                        'title': row['title'],
                        'created_at': row['created_at'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                        'username': row['username']
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get chat info: {e}")
            return None
    
    def search_chats(self, username: str, query: str, limit: int = 20) -> List[Dict]:
        """Search chats by title or message content"""
        try:
            user_id = self.ensure_user_exists(username)
            
            with self.get_db() as conn:
                rows = conn.execute("""
                    SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at
                    FROM chats c
                    LEFT JOIN messages m ON c.id = m.chat_id
                    WHERE c.user_id = ? AND c.is_active = 1
                    AND (c.title LIKE ? OR m.content LIKE ?)
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                """, (user_id, f'%{query}%', f'%{query}%', limit)).fetchall()
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to search chats: {e}")
            return []
