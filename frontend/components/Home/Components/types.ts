import { ReactNode } from "react";

// Define an IconConfig type to replace direct ReactNode JSX elements
export type IconConfig = {
  type: "icon";
  name: string;
  props?: {
    className?: string;
    [key: string]: any;
  };
};

export type Section = {
  title: string;
  description: string;
};

export type Link = {
  text: string;
  href: string;
  icon?: IconConfig;
};

export type HomePageConfig = {
  hero: {
    title: string;
    description: string;
    primaryCta: Link;
    secondaryCta: Link;
    imageUrl?: string;
  };
  howItWorks: Section & {
    steps: Array<{
      icon: IconConfig;
      title: string;
      description: string;
    }>;
    ctaText: string;
    ctaLink: string;
  };
  benefits: Section & {
    items: Array<{
      title: string;
      description: string;
    }>;
  };
  cta: Section & {
    primaryLink: Link & {
      variant?:
        | "default"
        | "outline"
        | "secondary"
        | "destructive"
        | "ghost"
        | "link";
    };
    secondaryLink: Link & {
      variant?:
        | "default"
        | "outline"
        | "secondary"
        | "destructive"
        | "ghost"
        | "link";
    };
  };
};
