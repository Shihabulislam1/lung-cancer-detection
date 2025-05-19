import { drizzle } from "drizzle-orm/postgres-js"
import postgres from "postgres"
import * as schema from "./schema"

// For local development with Docker
const connectionString =
  process.env.DATABASE_URL ||
  "postgres://myuser:mypassword@localhost:5432/cancer-database" // Updated fallback

// Connection for migrations
export const migrationClient = postgres(connectionString, { max: 1 })

// Connection for query builder
const queryClient = postgres(connectionString)
export const db = drizzle(queryClient, { schema })
