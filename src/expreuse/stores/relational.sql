-- Normalized relational realization.
-- Task is split into task and goal so that each distinct goal is embedded and
-- compared once, however many tasks and traces share it.
CREATE TABLE IF NOT EXISTS goal (
    goal_id     bigserial PRIMARY KEY,
    application text NOT NULL,
    goal        text NOT NULL,
    embedding   vector({dim}) NOT NULL,
    UNIQUE (application, goal)
);
-- Keep embeddings inline: exact search scans every goal, and fetching each
-- vector from TOAST would dominate the scan.
ALTER TABLE goal ALTER COLUMN embedding SET STORAGE PLAIN;
CREATE TABLE IF NOT EXISTS task (
    task_id     text PRIMARY KEY,
    application text NOT NULL,
    goal_id     bigint NOT NULL REFERENCES goal
);
CREATE TABLE IF NOT EXISTS trace (
    trace_id text PRIMARY KEY,
    task_id  text NOT NULL REFERENCES task,
    producer text NOT NULL,
    success  boolean NOT NULL,
    length   integer NOT NULL
);
CREATE TABLE IF NOT EXISTS subtrace (
    trace_id   text NOT NULL REFERENCES trace,
    subgoal_id integer NOT NULL,
    operator   text NOT NULL,
    binding    jsonb NOT NULL,
    finished   boolean NOT NULL,
    fits       boolean NOT NULL,
    length     integer NOT NULL,
    PRIMARY KEY (trace_id, subgoal_id)
);
CREATE TABLE IF NOT EXISTS step (
    trace_id    text NOT NULL REFERENCES trace,
    step_index  integer NOT NULL,
    subgoal_id  integer,
    action      text NOT NULL,
    observation text NOT NULL,
    PRIMARY KEY (trace_id, step_index)
);
CREATE INDEX IF NOT EXISTS task_goal_idx ON task (goal_id);
CREATE INDEX IF NOT EXISTS trace_task_idx ON trace (task_id, producer);
CREATE INDEX IF NOT EXISTS subtrace_operator_idx ON subtrace (trace_id, operator)
    WHERE finished AND fits;
