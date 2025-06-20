use auramark_engine::{image_handler, robust, utils};
use clap::{Parser, Subcommand};
use image::DynamicImage;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Embed {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long)]
        message: String,
        #[arg(short, long)]
        secret_key: String,
    },
    Extract {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        secret_key: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Embed {
            input,
            output,
            message,
            secret_key,
        } => {
            // Load image as DynamicImage
            let image = image_handler::load_image_from_bytes(&std::fs::read(input)?)?;
            let mut rgb_image = image.to_rgb8();
            let message_bytes = utils::bytes_utils::to_fixed_16(message.as_bytes());
            let secret_key_bytes = utils::bytes_utils::to_fixed_16(secret_key.as_bytes());

            // Embed watermark in-place
            robust::embed(&mut rgb_image, &message_bytes, &secret_key_bytes)?;

            // After embed returns, convert back to DynamicImage for saving
            let output_image = DynamicImage::ImageRgb8(rgb_image);

            // Save output image bytes
            let output_bytes = image_handler::save_image_to_bytes(&output_image)?;
            std::fs::write(output, output_bytes)?;
            println!("Embedding done.");
        }

        Commands::Extract { input, secret_key } => {
            let image = image_handler::load_image_from_bytes(&std::fs::read(&input)?)?;
            let secret_key_bytes = utils::bytes_utils::to_fixed_16(secret_key.as_bytes());
            match robust::extract(&image, &secret_key_bytes)? {
                Some(wm) => println!("{}", wm),
                None => eprintln!("No watermark found or invalid key."),
            }
        }
    }

    Ok(())
}
