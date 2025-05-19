"use client";

import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import DynamicIcon from "./DynamicIcon";
import { IconConfig } from "./types";

type HeroProps = {
  title: string;
  description: string;
  primaryCta: {
    text: string;
    href: string;
    icon?: IconConfig;
  };
  secondaryCta: {
    text: string;
    href: string;
  };
  imageUrl?: string;
};

const Hero: React.FC<HeroProps> = ({
  title,
  description,
  primaryCta,
  secondaryCta,
  imageUrl = "/placeholder.jpg",
}) => {
  return (
    <section className="py-24 px-6 bg-gradient-to-b from-background to-background/80">
      <motion.div
        className="container mx-auto max-w-6xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className="flex flex-col gap-8 md:flex-row items-center">
          <div className="flex-1 space-y-6">
            <motion.h1
              className="text-4xl md:text-6xl font-bold tracking-tight"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              {title}
            </motion.h1>
            <motion.p
              className="text-xl text-muted-foreground"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.8 }}
            >
              {description}
            </motion.p>
            <motion.div
              className="flex flex-wrap gap-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              {" "}
              <Button asChild size="lg">
                <Link href={primaryCta.href}>
                  {primaryCta.text}
                  {primaryCta.icon && <DynamicIcon config={primaryCta.icon} />}
                </Link>
              </Button>
              <Button variant="outline" size="lg" asChild>
                <Link href={secondaryCta.href}>{secondaryCta.text}</Link>
              </Button>
            </motion.div>
          </div>
          <motion.div
            className="flex-1"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4, duration: 0.8 }}
          >
            <div className="relative h-80 md:h-96 w-full rounded-xl bg-muted overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-tr from-primary/20 to-primary/5 z-10"></div>
              <div
                className="absolute inset-0 bg-[url('/placeholder.jpg')] bg-cover bg-center"
                style={{ backgroundImage: `url(${imageUrl})` }}
              ></div>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
};

export default Hero;
