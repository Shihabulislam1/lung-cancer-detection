import React from "react";
import Hero from "../Components/Hero";
import HowItWorks from "../Components/HowItWorks";
import Benefits from "../Components/Benefits";
import CtaSection from "../Components/CtaSection";
import homeConfig from "../Components/config";

const HomePage = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <Hero
        title={homeConfig.hero.title}
        description={homeConfig.hero.description}
        primaryCta={homeConfig.hero.primaryCta}
        secondaryCta={homeConfig.hero.secondaryCta}
        imageUrl={homeConfig.hero.imageUrl}
      />
      
      <HowItWorks
        title={homeConfig.howItWorks.title}
        description={homeConfig.howItWorks.description}
        steps={homeConfig.howItWorks.steps}
        ctaText={homeConfig.howItWorks.ctaText}
        ctaLink={homeConfig.howItWorks.ctaLink}
      />
      
      <Benefits
        title={homeConfig.benefits.title}
        description={homeConfig.benefits.description}
        benefits={homeConfig.benefits.items}
      />
      
      <CtaSection
        title={homeConfig.cta.title}
        description={homeConfig.cta.description}
        primaryLink={homeConfig.cta.primaryLink}
        secondaryLink={homeConfig.cta.secondaryLink}
      />
    </div>
  );
};

export default HomePage;
