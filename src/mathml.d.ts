// Native MathML elements (rendered by all current browsers) for the Lab's equations.
import "react";

type MathProps = React.HTMLAttributes<HTMLElement> & { display?: "block" | "inline" };

declare module "react" {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace JSX {
    interface IntrinsicElements {
      math: MathProps; mrow: MathProps; mi: MathProps; mo: MathProps; mn: MathProps;
      msub: MathProps; msup: MathProps; mfrac: MathProps;
    }
  }
}
