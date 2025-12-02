"""
Database operations for Sentiment Analysis Web App
"""
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import pandas as pd


class Database:
    def __init__(self, db_path: str = "sentiment_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Sentiment results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                review_text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                aspect TEXT,
                topic_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        
        # Manual annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS manual_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentiment_result_id INTEGER,
                corrected_sentiment TEXT NOT NULL,
                annotated_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentiment_result_id) REFERENCES sentiment_results(id),
                FOREIGN KEY (annotated_by) REFERENCES users(id)
            )
        """)
        
        # Model versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_number TEXT NOT NULL,
                accuracy REAL,
                training_samples INTEGER,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> Optional[int]:
        """Create a new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, role))
            conn.commit()
            user_id = cursor.lastrowid
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute("""
            SELECT id, username, email, role FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return dict(result)
        return None
    
    def create_project(self, name: str, description: str, user_id: int) -> int:
        """Create a new project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO projects (name, description, user_id)
            VALUES (?, ?, ?)
        """, (name, description, user_id))
        
        conn.commit()
        project_id = cursor.lastrowid
        conn.close()
        return project_id
    
    def get_user_projects(self, user_id: int) -> List[Dict]:
        """Get all projects for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, created_at FROM projects
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def insert_sentiment_results(self, project_id: int, results: List[Dict]):
        """Insert sentiment analysis results"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute("""
                INSERT INTO sentiment_results 
                (project_id, review_text, sentiment, confidence, aspect, topic_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                project_id,
                result['review_text'],
                result['sentiment'],
                result['confidence'],
                result.get('aspect'),
                result.get('topic_id')
            ))
        
        conn.commit()
        conn.close()
    
    def get_low_confidence_reviews(self, project_id: Optional[int] = None, threshold: float = 0.50) -> List[Dict]:
        """Get reviews with confidence below threshold"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if project_id:
            cursor.execute("""
                SELECT sr.id, sr.review_text, sr.sentiment, sr.confidence, sr.aspect, sr.created_at,
                       sr.project_id, p.name as project_name
                FROM sentiment_results sr
                LEFT JOIN projects p ON sr.project_id = p.id
                WHERE sr.confidence < ? AND sr.project_id = ?
                AND sr.id NOT IN (SELECT sentiment_result_id FROM manual_annotations WHERE sentiment_result_id IS NOT NULL)
                ORDER BY sr.confidence ASC
            """, (threshold, project_id))
        else:
            cursor.execute("""
                SELECT sr.id, sr.review_text, sr.sentiment, sr.confidence, sr.aspect, sr.created_at,
                       sr.project_id, p.name as project_name
                FROM sentiment_results sr
                LEFT JOIN projects p ON sr.project_id = p.id
                WHERE sr.confidence < ?
                AND sr.id NOT IN (SELECT sentiment_result_id FROM manual_annotations WHERE sentiment_result_id IS NOT NULL)
                ORDER BY sr.confidence ASC
            """, (threshold,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def save_manual_annotation(self, sentiment_result_id: int, corrected_sentiment: str, user_id: int):
        """Save manual annotation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO manual_annotations (sentiment_result_id, corrected_sentiment, annotated_by)
            VALUES (?, ?, ?)
        """, (sentiment_result_id, corrected_sentiment, user_id))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self, project_id: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Get training data (original + corrections)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if project_id:
            # Original data (excluding corrected ones)
            cursor.execute("""
                SELECT sr.review_text, sr.sentiment 
                FROM sentiment_results sr
                LEFT JOIN manual_annotations ma ON sr.id = ma.sentiment_result_id
                WHERE sr.project_id = ? AND ma.id IS NULL
            """, (project_id,))
            original_data = cursor.fetchall()
            
            # Corrected data
            cursor.execute("""
                SELECT sr.review_text, ma.corrected_sentiment as sentiment
                FROM manual_annotations ma
                JOIN sentiment_results sr ON ma.sentiment_result_id = sr.id
                WHERE sr.project_id = ?
            """, (project_id,))
            corrected_data = cursor.fetchall()
        else:
            # Original data (excluding corrected ones)
            cursor.execute("""
                SELECT sr.review_text, sr.sentiment 
                FROM sentiment_results sr
                LEFT JOIN manual_annotations ma ON sr.id = ma.sentiment_result_id
                WHERE ma.id IS NULL
            """)
            original_data = cursor.fetchall()
            
            # Corrected data
            cursor.execute("""
                SELECT sr.review_text, ma.corrected_sentiment as sentiment
                FROM manual_annotations ma
                JOIN sentiment_results sr ON ma.sentiment_result_id = sr.id
            """)
            corrected_data = cursor.fetchall()
        
        conn.close()
        
        # Combine data (corrections override originals)
        texts = []
        sentiments = []
        
        # Add originals
        for row in original_data:
            texts.append(row[0])
            sentiments.append(row[1])
        
        # Add corrections
        for row in corrected_data:
            texts.append(row[0])
            sentiments.append(row[1])
        
        return texts, sentiments
    
    def get_sentiment_stats(self, project_id: Optional[int] = None) -> Dict:
        """Get sentiment statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if project_id:
            cursor.execute("""
                SELECT sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM sentiment_results
                WHERE project_id = ?
                GROUP BY sentiment
            """, (project_id,))
        else:
            cursor.execute("""
                SELECT sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM sentiment_results
                GROUP BY sentiment
            """)
        
        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {'count': row[1], 'avg_confidence': row[2]}
        
        conn.close()
        return stats
    
    def get_sentiment_over_time(self, project_id: Optional[int] = None) -> pd.DataFrame:
        """Get sentiment distribution over time"""
        conn = self.get_connection()
        
        if project_id:
            query = """
                SELECT DATE(created_at) as date, sentiment, COUNT(*) as count
                FROM sentiment_results
                WHERE project_id = ?
                GROUP BY DATE(created_at), sentiment
                ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn, params=(project_id,))
        else:
            query = """
                SELECT DATE(created_at) as date, sentiment, COUNT(*) as count
                FROM sentiment_results
                GROUP BY DATE(created_at), sentiment
                ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_admin_stats(self) -> Dict:
        """Get admin dashboard statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Active users
        cursor.execute("SELECT COUNT(DISTINCT id) FROM users")
        active_users = cursor.fetchone()[0]
        
        # Total annotations
        cursor.execute("SELECT COUNT(*) FROM manual_annotations")
        total_annotations = cursor.fetchone()[0]
        
        # Pending uncertain samples
        cursor.execute("SELECT COUNT(*) FROM sentiment_results WHERE confidence < 0.50")
        pending_samples = cursor.fetchone()[0]
        
        # Model versions
        cursor.execute("SELECT COUNT(*) FROM model_versions")
        model_count = cursor.fetchone()[0]
        
        # Latest model accuracy
        cursor.execute("""
            SELECT accuracy FROM model_versions
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        model_accuracy = result[0] if result else None
        
        conn.close()
        
        return {
            'active_users': active_users,
            'total_annotations': total_annotations,
            'pending_samples': pending_samples,
            'model_count': model_count,
            'model_accuracy': model_accuracy
        }
    
    def save_model_version(self, version_number: str, accuracy: float, training_samples: int, model_path: str):
        """Save model version"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Deactivate old models
        cursor.execute("UPDATE model_versions SET is_active = 0")
        
        # Insert new model
        cursor.execute("""
            INSERT INTO model_versions (version_number, accuracy, training_samples, model_path, is_active)
            VALUES (?, ?, ?, ?, 1)
        """, (version_number, accuracy, training_samples, model_path))
        
        conn.commit()
        conn.close()
    
    def get_model_versions(self) -> List[Dict]:
        """Get all model versions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, version_number, accuracy, training_samples, model_path, created_at, is_active
            FROM model_versions
            ORDER BY created_at DESC
        """)
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

