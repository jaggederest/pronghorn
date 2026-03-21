use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    info!("pronghorn v{}", env!("CARGO_PKG_VERSION"));
}
