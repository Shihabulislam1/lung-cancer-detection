"use client";
import React from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import DynamicIcon from "./DynamicIcon";
import { IconConfig } from "./types";

export type Step = {
  icon: IconConfig;
  title: string;
  description: string;
};

type HowItWorksProps = {
  title: string;
  description: string;
  steps: Step[];
  ctaText: string;
  ctaLink: string;
};

const HowItWorks: React.FC<HowItWorksProps> = ({
  title,
  description,
  steps,
  ctaText,
  ctaLink,
}) => {
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 },
  };

  return (
    <section className="py-24 px-6 bg-muted/50">
      <div className="container mx-auto max-w-6xl">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4">
            {title}
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            {description}
          </p>
        </motion.div>

        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-8"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
        >
          {steps.map((step, index) => (
            <motion.div key={index} variants={itemVariants}>
              <Card className="h-full border-none shadow-lg hover:shadow-xl transition-shadow duration-300">
                <CardHeader className="pb-2">
                  <div className="mb-4 flex justify-center">
                    <DynamicIcon config={step.icon} />
                  </div>
                  <CardTitle className="text-xl text-center">
                    {step.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-center text-base">
                    {step.description}
                  </CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          className="mt-12 text-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.6, duration: 0.6 }}
        >
          <Button asChild size="lg">
            <Link href={ctaLink}>{ctaText}</Link>
          </Button>
        </motion.div>
      </div>
    </section>
  );
};

export default HowItWorks;
