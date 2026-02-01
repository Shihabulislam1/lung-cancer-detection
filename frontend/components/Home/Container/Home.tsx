import React from "react";
import Hero from "../Components/Hero";
import HowItWorks from "../Components/HowItWorks";
import Benefits from "../Components/Benefits";
import CtaSection from "../Components/CtaSection";
import homeConfig from "../Components/config";

const HomePage = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mx-auto container my-6 rounded">
        <h3 className="font-semibold">Beta Version - Development Notice</h3>
        <p className="text-sm mt-1">This AI tool is in active development. Results should NOT replace professional medical consultation. Always consult licensed healthcare providers for diagnosis.</p>
      </div>
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
      <section className="container mx-auto py-12">
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-6 border rounded">
            <h2 className="text-xl font-bold mb-2">About Project</h2>
            <p className="text-sm mb-4">This project — part of a Master's thesis at the Department of Mechatronics Engineering, Rajshahi University of Engineering & Technology (RUET) — develops an AI-based system for early detection of lung cancer from medical imaging data. It aims to assist healthcare professionals with preliminary screening using machine learning on CT scans.</p>
            <a href="/about" className="text-primary font-medium">Read full details →</a>
          </div>
          <div className="p-6 border rounded">
            <h2 className="text-xl font-bold mb-2">About Researcher</h2>
            <p className="text-sm mb-4">This research is conducted by Khandakar Rabbi Ahmed, a Master's thesis candidate at RUET. For inquiries or collaboration, contact the researcher.</p>
            <a href="/researcher" className="text-primary font-medium">Contact / Read more →</a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
