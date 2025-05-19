"use client";
import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Link from "next/link";

type CtaLink = {
  text: string;
  href: string;
  variant?:
    | "default"
    | "outline"
    | "secondary"
    | "destructive"
    | "ghost"
    | "link";
};

type CtaSectionProps = {
  title: string;
  description: string;
  primaryLink: CtaLink;
  secondaryLink?: CtaLink;
  bgClassName?: string;
};

const CtaSection: React.FC<CtaSectionProps> = ({
  title,
  description,
  primaryLink,
  secondaryLink,
  bgClassName = "bg-primary/5",
}) => {
  return (
    <section className={`py-16 px-6 ${bgClassName}`}>
      <motion.div
        className="container mx-auto max-w-4xl text-center"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
      >
        <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-6">
          {title}
        </h2>
        <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
          {description}
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          {" "}
          <Button
            asChild
            size="lg"
            variant={
              (primaryLink.variant as
                | "default"
                | "outline"
                | "secondary"
                | "destructive"
                | "ghost"
                | "link") || "default"
            }
          >
            <Link href={primaryLink.href}>{primaryLink.text}</Link>
          </Button>
          {secondaryLink && (
            <Button
              asChild
              size="lg"
              variant={
                (secondaryLink.variant as
                  | "default"
                  | "outline"
                  | "secondary"
                  | "destructive"
                  | "ghost"
                  | "link") || "outline"
              }
            >
              <Link href={secondaryLink.href}>{secondaryLink.text}</Link>
            </Button>
          )}
        </div>
      </motion.div>
    </section>
  );
};

export default CtaSection;
