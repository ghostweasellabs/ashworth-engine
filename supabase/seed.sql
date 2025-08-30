-- Ashworth Engine Database Seed Data
-- This file contains initial data for development and testing

-- Insert sample data for development
INSERT INTO workflow_runs (workflow_id, status, input_data) VALUES 
('sample-workflow-1', 'completed', '{"file_name": "sample.xlsx", "file_size": 1024}'),
('sample-workflow-2', 'pending', '{"file_name": "test.csv", "file_size": 2048}')
ON CONFLICT DO NOTHING;

-- Create a health check function
CREATE OR REPLACE FUNCTION health_check()
RETURNS TABLE(status TEXT, ts TIMESTAMP WITH TIME ZONE) AS $$
BEGIN
    RETURN QUERY SELECT 'healthy'::TEXT, NOW();
END;
$$ LANGUAGE plpgsql;