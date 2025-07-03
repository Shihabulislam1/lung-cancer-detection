CREATE TYPE "public"."report_status" AS ENUM('pending', 'completed', 'failed');--> statement-breakpoint
CREATE TABLE "cancer_reports" (
	"id" serial PRIMARY KEY NOT NULL,
	"user_id" integer NOT NULL,
	"image_url" text NOT NULL,
	"public_id" text,
	"status" "report_status" DEFAULT 'pending' NOT NULL,
	"raw_output" json,
	"probabilities" json,
	"predicted_class_index" integer,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" varchar(255) NOT NULL,
	"email" varchar(255) NOT NULL,
	"password" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
ALTER TABLE "cancer_reports" ADD CONSTRAINT "cancer_reports_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE no action ON UPDATE no action;