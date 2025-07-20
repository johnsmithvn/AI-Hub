"""
AI Backend Hub - Comprehensive Database Migration Script
Creates all necessary tables and indexes for the AI system
"""

# Database Migration - Create Tables
revision = '001'
down_revision = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('username', sa.String(50), nullable=False, unique=True),
        sa.Column('email', sa.String(100), nullable=True, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(100), nullable=True),
        sa.Column('role', sa.Enum('ADMIN', 'USER', 'GUEST', name='userrole'), nullable=False, default='USER'),
        sa.Column('preferences', sa.JSON, default={}),
        sa.Column('default_model', sa.String(100), nullable=True),
        sa.Column('api_key', sa.String(255), nullable=True, unique=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('is_verified', sa.Boolean, default=False),
    )
    
    # Create indexes for users
    op.create_index('ix_users_username', 'users', ['username'])
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_api_key', 'users', ['api_key'])

def downgrade():
    op.drop_table('users')
    # Add other table drops...
