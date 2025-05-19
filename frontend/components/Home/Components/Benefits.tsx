"use client";
import React from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export type Benefit = {
  title: string;
  description: string;
};

type BenefitsProps = {
  title: string;
  description: string;
  benefits: Benefit[];
};

const Benefits: React.FC<BenefitsProps> = ({
  title,
  description,
  benefits,
}) => {
  return (
    <section className="py-24 px-6">
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

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {benefits.map((benefit, index) => {
            // Alternate animation direction for each card
            const isEven = index % 2 === 0;

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: isEven ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{
                  delay: Math.floor(index / 2) * 0.2,
                  duration: 0.6,
                }}
              >
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>{benefit.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p>{benefit.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Benefits;
