import os
import subprocess
import numpy as np
import pandas as pd
import shutil

"""
Document Manager class to generate documentation for each simulation run.

TODO LIST:
- Change it to, rather than load material data from .csvs, load from objects themselves.
    Cases where this stops a bug should be rare but would be nice
- Add image of geometry to title page

"""

class document_manager:
    def __init__(self, analysis_manager, label, workspace_path = "/workspace"):
        print("Created document manager")
        self.workspace_path = workspace_path
        self.label = label.replace("_", r"\_").replace("*", r"\*")
        self.am = analysis_manager
        self.experiment_name = self.am.experiment_name
        self.escaped_experiment_name = self.am.experiment_name.replace("_", r"\_").replace("*", r"\*")

        os.makedirs(f"{self.workspace_path}/results/{self.experiment_name}-{self.label}", exist_ok=True)
        os.makedirs(f"{self.workspace_path}/results/{self.experiment_name}-{self.label}/plots", exist_ok=True)
        shutil.copy2(f"{self.workspace_path}/results/template.typ", f"{self.workspace_path}/results/{self.experiment_name}-{self.label}")
        self.am.plot_dir = f"{self.workspace_path}/results/{self.experiment_name}-{self.label}/plots"
        self.am.execute_plots()

    def generate_typst_file(self):
        """
        Generate a Typst file based on the analysis_manager data.
        """
        shutil.copyfile(f"{self.workspace_path}/template.typ", f"{self.workspace_path}/results/{self.experiment_name}-{self.label}/template.typ")
        filename = f"{self.workspace_path}/results/{self.experiment_name}-{self.label}/report.typ"
        content = self._create_typst_content()

        with open(filename, "w") as f:
            f.write(content)

        print(f"Typst file '{filename}' has been generated.")

    def _create_typst_content(self):
        content = f"""
#import "template.typ": *

#show: project.with(
  title: [Chroma Simulation \ {self.escaped_experiment_name} \ {self.label}],
  authors: ((name: "Pocar Lab", email: "Lab 21", affiliation: "nEXO Collaboration"),),
  date: [#datetime.today().display()],
)

= Experiment Details

- Experiment Name: {self.escaped_experiment_name}
- Number of Particles: {self.am.num_particles}
- Random Seed: {self.am.seed}
- Excluded Geometry: {self.am.gm.exclude}
- PTE: {self.am.photon_transport_efficiency:.5g} $plus.minus$ {self.am.pte_st_dev:.2g}.

#pagebreak()
= Material Properties

== Bulk Materials

{self._generate_bulk_materials_table()}

== Surface Properties

{self._generate_surface_props_table()}

= Results

== Photon Transport Efficiency

The Photon Transport Efficiency (PTE) for this experiment was 
{self.am.photon_transport_efficiency:.5g} $plus.minus$ {self.am.pte_st_dev:.2g}.

== Tallies

#table(
  columns: (auto, auto),
  [Tally], [Count],
  {self._generate_tally_rows()}
)
#pagebreak()
= Plots

#let plot_files = {self._get_plot_files()}

{self._generate_plot_grid()}

"""
        return content

    def _generate_plot_grid(self):
        return """
#grid(
  columns: 2,
  gutter: 10pt,
  ..plot_files.map(file => {{
    figure(
      image(file, width: 100%),
      caption: file.split("/").at(-1).replace("_", " ").replace(".png", "")
    )
  }})
)
""".format()  # Using .format() to interpret the Typst curly braces literally

    def _generate_bulk_materials_table(self):
        bulk_materials_path = f"/workspace/data_files/data/{self.experiment_name}/bulk_materials_{self.experiment_name}.csv"
        df = pd.read_csv(bulk_materials_path)

        # Convert numeric columns to scientific notation
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                # df[col] = df[col].apply(lambda x: f"{x:.2e}")
                # df[col] = df[col].apply(format_number)
                df[col] = df[col].apply(lambda x: f"{x}")

        # Split columns into two groups
        name_column = df.columns[0]
        data_columns = df.columns[1:]
        mid_point = len(data_columns) // 2
        first_half = data_columns[:mid_point]
        second_half = data_columns[mid_point:]

        # Generate the first table
        table_content = "#table(\n  columns: " + str(len(first_half) + 1) + ",\n"
        table_content += (
            f"  [{name_column}], "
            + ", ".join([f"[{col}]" for col in first_half])
            + ",\n"
        )

        for _, row in df.iterrows():
            table_content += (
                f"  [{row[name_column]}], "
                + ", ".join([f"[{row[col]}]" for col in first_half])
                + ",\n"
            )

        table_content += ")\n\n"

        # Generate the second table
        table_content += "#table(\n  columns: " + str(len(second_half) + 1) + ",\n"
        table_content += (
            f"  [{name_column}], "
            + ", ".join([f"[{col}]" for col in second_half])
            + ",\n"
        )

        for _, row in df.iterrows():
            table_content += (
                f"  [{row[name_column]}], "
                + ", ".join([f"[{row[col]}]" for col in second_half])
                + ",\n"
            )

        table_content += ")"
        return table_content

    def _generate_surface_props_table(self):
        surface_props_path = f"/workspace/data_files/data/{self.experiment_name}/surface_props_{self.experiment_name}.csv"
        df = pd.read_csv(surface_props_path)

        table_content = "#table(\n  columns: " + str(len(df.columns)) + ",\n"
        table_content += "  " + ", ".join([f"[{col}]" for col in df.columns]) + ",\n"

        for _, row in df.iterrows():
            table_content += "  " + ", ".join([f"[{value}]" for value in row]) + ",\n"

        table_content += ")"
        return table_content

    def _generate_tally_rows(self):
        """
        Generate rows for the tally table.
        """
        rows = ""
        for key, value in self.am.tallies.items():
            rows += f"[{key}], [{np.sum(value)}],\n"
        return ""
        return rows

    def _get_plot_files(self):
        plot_dir = f"plots"  # Relative path from the Typst file location
        full_plot_dir = f"{self.workspace_path}/results/{self.experiment_name}-{self.label}/plots"
        plot_files = [f for f in os.listdir(full_plot_dir) if f.endswith(".png")]
        # Format as a Typst array with proper escaping
        return (
            "("
            + ", ".join(
                [
                    '"' + os.path.join(plot_dir, file).replace('"', '\\"') + '"'
                    for file in plot_files
                ]
            )
            + ")"
        )

    def compile_to_pdf(self, typst_file="report.typ", output_file="report.pdf"):
        """
        Compile the Typst file to PDF using the Typst compiler.
        """
        try:
            typst_file = f"{self.workspace_path}/results/{self.experiment_name}-{self.label}/report.typ"
            output_file = f"{self.workspace_path}/results/{self.experiment_name}-{self.label}/{self.experiment_name}-{self.label}.pdf"
            subprocess.run(["/workspace/typst", "compile", typst_file, output_file], check=True)
            print(f"PDF '{output_file}' has been generated.")
        except subprocess.CalledProcessError:
            print(
                "Error: Failed to compile Typst file. Make sure Typst is installed and accessible in your PATH."
            )
        except FileNotFoundError:
            print(
                "Error: Typst compiler not found. Please install Typst and make sure it's in your PATH."
            )
    def compile(self, output_file="report.pdf"):
        self.generate_typst_file()
        self.compile_to_pdf(output_file=output_file)
      
