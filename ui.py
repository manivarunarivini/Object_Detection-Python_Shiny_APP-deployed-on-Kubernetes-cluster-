from shiny import ui

CLASS_NAMES = [
    "Ivy", "Fern", "Ginkgo", "Kummerowia",
    "Laciniata", "Macrolobium",
    "Micranthes", "Murraya",
    "Robinia", "Selaginella"
]

app_ui = ui.page_sidebar(
    # Sidebar content
    ui.sidebar(
        ui.h4("Filters"),
        ui.input_select(
            id="class_filter",
            label="Select Leaf Class:",
            choices=["All"] + CLASS_NAMES,
            selected="All"
        ),
        
        #ui.hr(),
        ui.h4("Camera Configuration"),
        ui.input_text("resolution", "Resolution", ""),
        ui.input_numeric("measurement_interval", "Measurement Interval (sec)", None),
        ui.input_action_button("update", "Update Config"),
        ui.output_text("status"),
        #ui.hr(),
        ui.input_action_button("refresh_predictions", "Refresh Predictions", class_="btn-primary"),
        width=350  # Sidebar width in pixels
    ),
    
    # Main content
    ui.h2("Leaf Classification Results"),
    ui.output_ui("predicted_gallery"),
    ui.output_text("filter_stats"),
    
    # Existing JavaScript handlers
    ui.tags.script("""
        Shiny.addCustomMessageHandler("update_inputs", function(data) {
            $("#resolution").val(data.resolution);
            $("#measurement_interval").val(data.measurement_interval);
        });
    """),
    ui.tags.script("""
        Shiny.addCustomMessageHandler("status_update", function(message) {
            alert(message);
        });
    """)
)

