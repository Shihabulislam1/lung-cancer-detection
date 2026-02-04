import React from "react";

export default function ResearcherPage() {
  return (
    <div className="container mx-auto py-10">
      <h1 className="text-3xl font-bold mb-4">About the Researcher</h1>
      <p className="mb-4">
        This research project is being conducted by{" "}
        <strong>Md Shihabul Islam</strong>, BSc in Engineering in Mechatronics
        Engineering from Rajshahi University of Engineering & Technology (RUET).
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">Contact</h2>
      <p className="mb-4">
        For further inquiries, additional information, or collaboration
        opportunities related to the research, please contact:
      </p>
      <p className="font-medium">
        Email:{" "}
        <a
          href="mailto:mdshihabulislam.mte.ruet@gmail.com"
          className="text-primary"
        >
          mdshihabulislam.mte.ruet@gmail.com
        </a>
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">Research Focus</h2>
      <p className="mb-4">
        The research focuses on applying machine learning techniques to medical
        imaging for early lung cancer detection, with emphasis on precision,
        reliability, and clinical relevance.
      </p>
    </div>
  );
}
