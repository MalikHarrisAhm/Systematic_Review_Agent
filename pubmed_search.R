# Install and load required packages
if (!require("pubmedR")) {
    install.packages("devtools")
    devtools::install_github("massimoaria/pubmedR")
}
if (!require("bibliometrix")) {
    install.packages("bibliometrix")
}

library(pubmedR)
library(bibliometrix)

# Read search terms from file
search_terms <- readLines("search_terms.txt")

# Function to get data for a specific year range
get_year_data <- function(start_year, end_year, base_query, api_key) {
    year_query <- paste0(base_query, " AND (", start_year, ":", end_year, "[DP])")
    
    # Check count for this year range
    res <- pmQueryTotalCount(query = year_query, api_key = api_key)
    print(paste("Documents for", start_year, "-", end_year, ":", res$total_count))
    
    # Download data if there are results
    if (res$total_count > 0) {
        D <- pmApiRequest(query = year_query, limit = res$total_count, api_key = api_key)
        return(D)
    }
    return(NULL)
}

# Initialize list to store results
all_data <- list()
current_year <- as.numeric(format(Sys.Date(), "%Y"))

# Define year ranges
year_ranges <- list(
    c(2022, 2023),
    c(2024, 2025)
)

# Collect data for each year range
for (range in year_ranges) {
    period <- paste(range[1], range[2], sep="-")
    print(sprintf("\nProcessing years %s", period))
    data <- get_year_data(range[1], range[2], search_terms, api_key)
    if (!is.null(data)) {
        all_data[[length(all_data) + 1]] <- data
    }
    Sys.sleep(0.5)  # Add delay to respect API rate limits
}

# Combine all data
if (length(all_data) > 0) {
    combined_data <- combine_pubmed_data(all_data, search_terms)
    
    # Process and deduplicate
    processed_data <- process_and_deduplicate(combined_data)
    
    # Save results
    write.csv(processed_data$data_frame, "pubmed_search_results.csv", row.names = FALSE)
    saveRDS(processed_data$bibliometrix_df, "pubmed_search_bibliometrix.rds")
    saveRDS(processed_data$analysis, "pubmed_search_analysis.rds")
    
    print("Search completed successfully!")
} else {
    print("No results found.")
} 