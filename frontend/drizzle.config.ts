import type { Config } from "drizzle-kit";

export default {
  schema: "./lib/db/schema.ts",
  out: "./lib/db/migration",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL,
  },
} satisfies Config;
