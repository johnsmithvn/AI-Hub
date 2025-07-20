-- Initialize AI Hub Database
-- This script sets up the initial database schema and extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create initial admin user (password: admin123)
-- In production, change this password immediately
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN
        -- This will be created by Alembic migrations
        RAISE NOTICE 'Tables will be created by Alembic migrations';
    END IF;
END $$;
