import pg from "pg";
import dotenv from "dotenv";

dotenv.config();

const { Pool } = pg;

export const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

pool.on("error", (err) => {
  console.error("[db] Unexpected error on idle PostgreSQL client:", err);
});

export async function testConnection() {
  const client = await pool.connect();
  try {
    const result = await client.query("SELECT NOW()");
    console.log(`[db] Connected to PostgreSQL. Server time: ${result.rows[0].now}`);
  } finally {
    client.release();
  }
}