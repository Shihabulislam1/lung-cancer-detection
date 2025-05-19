import { migrate } from "drizzle-orm/postgres-js/migrator"
import { db } from "@/lib/db"

async function main() {
  console.log("Migration started...")
  await migrate(db, { migrationsFolder: "./drizzle" })
  console.log("Migration completed!")
  process.exit(0)
}

main().catch((error) => {
  console.error("Migration failed:", error)
  process.exit(1)
})
