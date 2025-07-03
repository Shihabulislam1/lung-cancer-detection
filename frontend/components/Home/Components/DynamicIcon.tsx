"use client";

import React from "react";
import * as LucideIcons from "lucide-react";
import { IconConfig } from "./types";

type DynamicIconProps = {
  config: IconConfig;
};

const DynamicIcon: React.FC<DynamicIconProps> = ({ config }) => {
  const { name, props } = config;
  const Icon = (
    LucideIcons as unknown as Record<
      string,
      React.ComponentType<React.SVGProps<SVGSVGElement>>
    >
  )[name];

  if (!Icon) {
    console.warn(`Icon "${name}" not found in Lucide icons`);
    return null;
  }

  return <Icon {...props} />;
};

export default DynamicIcon;
