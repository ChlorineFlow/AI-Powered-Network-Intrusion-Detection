/**
 * Run schema.sql against the configured database. Run manually once
 * (and again any time schema.sql changes) — this is not run
 * automatically on every server start.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { pool, testConnection } from "../config/db.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function migrate() {
  await testConnection();

  const schemaPath = path.join(__dirname, "schema.sql");
  const schemaSql = fs.readFileSync(schemaPath, "utf-8");

  console.log("[migrate] Applying schema.sql...");
  await pool.query(schemaSql);
  console.log("[migrate] Done. Tables created (or already existed): predictions, alerts.");

  await pool.end();
}

migrate().catch((err) => {
  console.error("[migrate] Failed:", err);
  process.exit(1);
});