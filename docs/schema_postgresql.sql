-- PostgreSQL schema of the Acko AI Platform: 4 tables.
-- Made by `python -m scripts.export_schema` from the models in acko/db.py.
-- `python -m scripts.init_db` creates the same tables (SQLite on a laptop, PostgreSQL on AWS RDS).

CREATE TABLE users (
    id UUID NOT NULL,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL,
    phone VARCHAR(15),
    role VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_users_email ON users (email);

CREATE TABLE quotations (
    id UUID NOT NULL,
    user_id UUID NOT NULL,
    vehicle_type VARCHAR(10) NOT NULL,
    vehicle_make VARCHAR(50) NOT NULL,
    vehicle_model VARCHAR(80) NOT NULL,
    manufacturing_year INTEGER NOT NULL,
    city VARCHAR(80) NOT NULL,
    idv FLOAT NOT NULL,
    ncb_percent INTEGER NOT NULL,
    predicted_premium FLOAT NOT NULL,
    fuel_type VARCHAR(20),
    policy_type VARCHAR(30),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_quotations_created_at ON quotations (created_at);
CREATE INDEX ix_quotations_user_id ON quotations (user_id);

CREATE TABLE claims (
    id UUID NOT NULL,
    user_id UUID NOT NULL,
    vehicle_type VARCHAR(10) NOT NULL,
    policy_number VARCHAR(50),
    incident_date DATE,
    damage_type VARCHAR(50),
    affected_parts VARCHAR(200),
    damage_severity VARCHAR(20),
    image_s3_key VARCHAR(255),
    form_s3_key VARCHAR(255),
    predicted_amount FLOAT NOT NULL,
    approval_probability FLOAT NOT NULL,
    status VARCHAR(20) NOT NULL,
    city VARCHAR(80),
    vehicle_make VARCHAR(50),
    vehicle_model VARCHAR(80),
    idv FLOAT,
    vehicle_registration VARCHAR(20),
    damage_description VARCHAR(500),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_claims_created_at ON claims (created_at);
CREATE INDEX ix_claims_user_id ON claims (user_id);

CREATE TABLE chat_logs (
    id UUID NOT NULL,
    user_id UUID NOT NULL,
    intent VARCHAR(30) NOT NULL,
    question TEXT NOT NULL,
    retrieved_source VARCHAR(255),
    response TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_chat_logs_created_at ON chat_logs (created_at);
CREATE INDEX ix_chat_logs_user_id ON chat_logs (user_id);
