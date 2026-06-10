document$.subscribe(async () => {
  if (typeof mermaid === "undefined") {
    return;
  }

  try {
    mermaid.initialize({ startOnLoad: false });
    await mermaid.run({ querySelector: ".mermaid" });
  } catch (error) {
    console.error("Mermaid rendering failed", error);
  }
});
