-- Setup data cleaning logs table for audit trail
CREATE TABLE public.data_cleaning_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id uuid REFERENCES public.analyses(id) ON DELETE CASCADE,
    operation text NOT NULL,
    input_data jsonb,
    output_data jsonb,
    quality_score decimal(5,2),
    issues_detected text[],
    created_at timestamptz DEFAULT now()
);

-- Create index for better query performance
CREATE INDEX ON public.data_cleaning_logs (analysis_id);
CREATE INDEX ON public.data_cleaning_logs (created_at);

-- RLS policy for data cleaning logs
ALTER TABLE public.data_cleaning_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role can manage all data cleaning logs" ON public.data_cleaning_logs
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');