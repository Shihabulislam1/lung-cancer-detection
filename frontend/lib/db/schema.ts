import { pgTable, serial, text, varchar, timestamp, integer, json, pgEnum } from "drizzle-orm/pg-core"
import type { InferSelectModel, InferInsertModel } from "drizzle-orm"
import { relations } from "drizzle-orm"

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 255 }).notNull(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  password: text("password").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
})

export const usersRelations = relations(users, ({ many }) => ({
  reports: many(cancerReports),
}))

export const reportStatusEnum = pgEnum("report_status", ["pending", "completed", "failed"])

export const cancerReports = pgTable("cancer_reports", {
  id: serial("id").primaryKey(),
  userId: integer("user_id")
    .notNull()
    .references(() => users.id),
  imageUrl: text("image_url").notNull(),
  publicId: text("public_id"),
  status: reportStatusEnum("status").default("pending").notNull(),
  rawOutput: json("raw_output").$type<number[][]>(),
  probabilities: json("probabilities").$type<number[][]>(),
  predictedClassIndex: integer("predicted_class_index"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
})

export const cancerReportsRelations = relations(cancerReports, ({ one }) => ({
  user: one(users, {
    fields: [cancerReports.userId],
    references: [users.id],
  }),
}))

export type User = InferSelectModel<typeof users>
export type NewUser = InferInsertModel<typeof users>

export type CancerReport = InferSelectModel<typeof cancerReports>
export type NewCancerReport = InferInsertModel<typeof cancerReports>
