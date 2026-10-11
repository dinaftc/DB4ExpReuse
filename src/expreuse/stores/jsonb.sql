CREATE TABLE IF NOT EXISTS goal (
    goal_id     bigserial PRIMARY KEY,
    application text NOT NULL,
    goal        text NOT NULL,
    embedding   vector({dim}) NOT NULL,
    UNIQUE (application, goal)
);
-- inline storage: exact search scans every goal, and TOAST fetches would dominate it
ALTER TABLE goal ALTER COLUMN embedding SET STORAGE PLAIN;
CREATE TABLE IF NOT EXISTS trace_doc (
    trace_id text PRIMARY KEY,
    task_id  text NOT NULL,
    goal_id  bigint NOT NULL REFERENCES goal,
    producer text NOT NULL,
    success  boolean NOT NULL,
    length   integer NOT NULL,
    -- operators with a finished, fitting subtrace
    operators text[] NOT NULL,
    -- {"subtraces": [{subgoal_id, operator, binding, finished, fits, length,
    --                 steps: [{step_index, action, observation}]}],
    --  "other_steps": [{step_index, action, observation}]}
    doc      jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS trace_doc_goal_idx ON trace_doc (goal_id, producer);
