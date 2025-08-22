-- Setup storage buckets for reports and charts
INSERT INTO storage.buckets (id, name, public) 
VALUES 
  ('reports', 'reports', false),
  ('charts', 'charts', false)
ON CONFLICT (id) DO NOTHING;