import { HomePageConfig } from "./types";

// We'll use icon name strings instead of direct JSX elements
// These will be converted to React components in the Home component
const homeConfig: HomePageConfig = {
  hero: {
    title: "Early Detection Saves Lives",
    description:
      "Our AI-powered lung cancer detection technology helps identify potential issues earlier with greater accuracy.",
    primaryCta: {
      text: "Try Detection Tool",
      href: "/cancer-detector",
      icon: {
        type: "icon",
        name: "ArrowRight",
        props: { className: "ml-2 h-4 w-4" },
      },
    },
    secondaryCta: {
      text: "Learn More",
      href: "#how-it-works",
    },
    imageUrl: "/placeholder.jpg",
  },
  howItWorks: {
    title: "How It Works",
    description:
      "Our lung cancer detection platform uses state-of-the-art AI to analyze CT scans and provide accurate results in three simple steps.",
    steps: [
      {
        icon: {
          type: "icon",
          name: "Upload",
          props: { className: "h-12 w-12 text-primary" },
        },
        title: "Upload CT Scan",
        description:
          "Upload your lung CT scan images through our secure platform.",
      },
      {
        icon: {
          type: "icon",
          name: "FileText",
          props: { className: "h-12 w-12 text-primary" },
        },
        title: "AI Analysis",
        description:
          "Our advanced AI model analyzes the images to detect potential signs of lung cancer.",
      },
      {
        icon: {
          type: "icon",
          name: "CheckCircle",
          props: { className: "h-12 w-12 text-primary" },
        },
        title: "Get Results",
        description:
          "Receive detailed analysis reports with visualizations and recommendations.",
      },
    ],
    ctaText: "Get Started",
    ctaLink: "/cancer-detector",
  },
  benefits: {
    title: "Why Choose Our Platform",
    description:
      "Our AI-powered detection system offers several advantages over traditional methods.",
    items: [
      {
        title: "Early Detection",
        description:
          "Our AI model can detect subtle patterns that might be missed in early stages, potentially saving lives through timely intervention.",
      },
      {
        title: "High Accuracy",
        description:
          "Trained on thousands of verified cases, our model provides high accuracy detection with minimal false positives.",
      },
      {
        title: "Fast Results",
        description:
          "Get your analysis results within minutes instead of waiting days for traditional radiologist reviews.",
      },
      {
        title: "Secure & Private",
        description:
          "All your medical data is encrypted and handled with strict privacy protocols to ensure confidentiality.",
      },
    ],
  },
  cta: {
    title: "Ready to take control of your health?",
    description:
      "Join thousands of users who have already benefited from our early detection technology.",
    primaryLink: {
      text: "Try Our Detection Tool",
      href: "/cancer-detector",
    },
    secondaryLink: {
      text: "View Dashboard",
      href: "/dashboard",
      variant: "outline",
    },
  },
};

export default homeConfig;
