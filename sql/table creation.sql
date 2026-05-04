CREATE TABLE IF NOT EXISTS circuits (
    circuit_id      TEXT        PRIMARY KEY,            
    name            TEXT        NOT NULL,               
    location        TEXT,                               
    country         TEXT,
    latitude        NUMERIC(9,6),
    longitude       NUMERIC(9,6),
    url             TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drivers (
    driver_id           TEXT        PRIMARY KEY,        
    permanent_number    SMALLINT,
    code                CHAR(3),                        
    forename            TEXT        NOT NULL,
    surname             TEXT        NOT NULL,
    date_of_birth       DATE,
    nationality         TEXT,
    url                 TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS constructors (
    constructor_id  TEXT        PRIMARY KEY,            
    name            TEXT        NOT NULL,
    nationality     TEXT,
    url             TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS races (
    race_id         SERIAL      PRIMARY KEY,
    season          SMALLINT    NOT NULL,
    round           SMALLINT    NOT NULL,
    circuit_id      TEXT        NOT NULL REFERENCES circuits(circuit_id),
    name            TEXT        NOT NULL,               
    race_date       DATE,
    race_time       TIME,
    url             TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_races_season_round UNIQUE (season, round)
);

CREATE INDEX IF NOT EXISTS idx_races_season     ON races (season);
CREATE INDEX IF NOT EXISTS idx_races_circuit_id ON races (circuit_id);


CREATE TABLE IF NOT EXISTS results (
    result_id           SERIAL      PRIMARY KEY,
    race_id             INT         NOT NULL REFERENCES races(race_id),
    driver_id           TEXT        NOT NULL REFERENCES drivers(driver_id),
    constructor_id      TEXT        NOT NULL REFERENCES constructors(constructor_id),
    grid                SMALLINT,                       
    position            SMALLINT,                       
    position_text       TEXT,                           
    position_order      SMALLINT    NOT NULL,           
    points              NUMERIC(5,2) NOT NULL DEFAULT 0,
    laps                SMALLINT,
    time_millis         INT,                            
    fastest_lap_rank    SMALLINT,                       
    fastest_lap_lap     SMALLINT,
    fastest_lap_time    TEXT,                           
    fastest_lap_speed   NUMERIC(7,3),                   
    status              TEXT,                           
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_results_race_driver UNIQUE (race_id, driver_id)
);

CREATE INDEX IF NOT EXISTS idx_results_race_id        ON results (race_id);
CREATE INDEX IF NOT EXISTS idx_results_driver_id      ON results (driver_id);
CREATE INDEX IF NOT EXISTS idx_results_constructor_id ON results (constructor_id);

CREATE TABLE IF NOT EXISTS qualifying (
    qualifying_id   SERIAL      PRIMARY KEY,
    race_id         INT         NOT NULL REFERENCES races(race_id),
    driver_id       TEXT        NOT NULL REFERENCES drivers(driver_id),
    constructor_id  TEXT        NOT NULL REFERENCES constructors(constructor_id),
    number          SMALLINT,                           
    position        SMALLINT,                           
    q1              TEXT,                               
    q2              TEXT,
    q3              TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_qualifying_race_driver UNIQUE (race_id, driver_id)
);

CREATE INDEX IF NOT EXISTS idx_qualifying_race_id   ON qualifying (race_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_driver_id ON qualifying (driver_id);

CREATE TABLE IF NOT EXISTS pit_stops (
    pit_stop_id     SERIAL      PRIMARY KEY,
    race_id         INT         NOT NULL REFERENCES races(race_id),
    driver_id       TEXT        NOT NULL REFERENCES drivers(driver_id),
    stop            SMALLINT    NOT NULL,               
    lap             SMALLINT    NOT NULL,
    local_time      TIME,                               
    duration_text   TEXT,                               
    duration_millis INT,                                

    CONSTRAINT uq_pit_stops_race_driver_stop UNIQUE (race_id, driver_id, stop)
);

CREATE INDEX IF NOT EXISTS idx_pit_stops_race_id   ON pit_stops (race_id);
CREATE INDEX IF NOT EXISTS idx_pit_stops_driver_id ON pit_stops (driver_id);

CREATE TABLE IF NOT EXISTS lap_times (
    lap_time_id     BIGSERIAL   PRIMARY KEY,
    race_id         INT         NOT NULL REFERENCES races(race_id),
    driver_id       TEXT        NOT NULL REFERENCES drivers(driver_id),
    lap             SMALLINT    NOT NULL,
    position        SMALLINT,                           
    time_text       TEXT,                               
    time_millis     INT,                                

    CONSTRAINT uq_lap_times_race_driver_lap UNIQUE (race_id, driver_id, lap)
);

CREATE INDEX IF NOT EXISTS idx_lap_times_race_id   ON lap_times (race_id);
CREATE INDEX IF NOT EXISTS idx_lap_times_driver_id ON lap_times (driver_id);




CREATE TABLE IF NOT EXISTS driver_standings (
    standing_id     SERIAL      PRIMARY KEY,
    race_id         INT         NOT NULL REFERENCES races(race_id),
    driver_id       TEXT        NOT NULL REFERENCES drivers(driver_id),
    constructor_id  TEXT        NOT NULL REFERENCES constructors(constructor_id),
    points          NUMERIC(6,2) NOT NULL DEFAULT 0,
    position        SMALLINT    NOT NULL,
    wins            SMALLINT    NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_driver_standings_race_driver UNIQUE (race_id, driver_id)
);

CREATE INDEX IF NOT EXISTS idx_driver_standings_race_id   ON driver_standings (race_id);
CREATE INDEX IF NOT EXISTS idx_driver_standings_driver_id ON driver_standings (driver_id);

CREATE TABLE IF NOT EXISTS constructor_standings (
    standing_id     SERIAL      PRIMARY KEY,
    race_id         INT         NOT NULL REFERENCES races(race_id),
    constructor_id  TEXT        NOT NULL REFERENCES constructors(constructor_id),
    points          NUMERIC(6,2) NOT NULL DEFAULT 0,
    position        SMALLINT    NOT NULL,
    wins            SMALLINT    NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_constructor_standings_race_constructor UNIQUE (race_id, constructor_id)
);

CREATE INDEX IF NOT EXISTS idx_constructor_standings_race_id    ON constructor_standings (race_id);
CREATE INDEX IF NOT EXISTS idx_constructor_standings_constructor ON constructor_standings (constructor_id);


CREATE TABLE IF NOT EXISTS ingest_log (
    log_id          SERIAL      PRIMARY KEY,
    endpoint        TEXT        NOT NULL,               
    season          SMALLINT,
    round           SMALLINT,
    status          TEXT        NOT NULL
                    CHECK (status IN ('success', 'error')),
    rows_upserted   INT,
    error_message   TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingest_log_endpoint ON ingest_log (endpoint, season, round);

CREATE TABLE IF NOT EXISTS fastf1_laps (
    lap_id          BIGSERIAL   PRIMARY KEY,
    race_id         INT         NOT NULL REFERENCES races(race_id),
    driver_code     TEXT        NOT NULL,                  
    lap_number      SMALLINT    NOT NULL,
    lap_time_ms     INT,                                   
    sector1_ms      INT,
    sector2_ms      INT,
    sector3_ms      INT,
    compound        TEXT,                                  
    tyre_life       SMALLINT,                              
    stint           SMALLINT,                              
    is_personal_best BOOLEAN    DEFAULT FALSE,
    track_status    TEXT,                                  
    deleted         BOOLEAN     DEFAULT FALSE,             
    CONSTRAINT uq_fastf1_laps UNIQUE (race_id, driver_code, lap_number)
);

CREATE INDEX IF NOT EXISTS idx_fastf1_laps_race_id     ON fastf1_laps (race_id);
CREATE INDEX IF NOT EXISTS idx_fastf1_laps_driver_code ON fastf1_laps (driver_code);

CREATE TABLE IF NOT EXISTS fastf1_weather (
    weather_id      SERIAL      PRIMARY KEY,
    race_id         INT         NOT NULL REFERENCES races(race_id),
    session_type    TEXT        NOT NULL,                  
    time_ms         INT,                                   
    air_temp        NUMERIC(5,2),
    track_temp      NUMERIC(5,2),
    humidity        NUMERIC(5,2),
    pressure        NUMERIC(7,2),
    wind_speed      NUMERIC(5,2),
    wind_direction  SMALLINT,
    rainfall        BOOLEAN     DEFAULT FALSE,
    CONSTRAINT uq_fastf1_weather UNIQUE (race_id, session_type, time_ms)
);

CREATE INDEX IF NOT EXISTS idx_fastf1_weather_race_id ON fastf1_weather (race_id);