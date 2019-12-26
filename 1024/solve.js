const readline = require('readline')
const fs = require('fs')
let matrix = []
let commands = []

function readFile (filePath, callback) {
  return new Promise((resolve, reject) => {
    let rl = readline.createInterface({
      input: fs.createReadStream(filePath),
      output: process.stdout,
      terminal: false
    })
    rl.on('line', line => {
      callback(line)
    })
    rl.on('close', line => {
      rl = null
      resolve(callback(null))
    })
  })
}

async function handleMatrix (data) {
  if (data) {
    matrix.push(data)
  } else {
    matrix = matrix.map(item => item.split(''))
    return matrix
  }
}

async function handleCommands (data) {
  if (data) {
    commands.push(data)
  } else {
    commands = commands.join(';').split(';').map(step => {
      let steps = step.replace(')', '').split('(')
      if (steps.length === 1) {
        steps.push(1)
      } else {
        steps[1] = parseInt(steps[1])
      }
      return steps
    })
    return commands
  }
}

function reverseDirection (name) {
  switch (name) {
    case 'up':
      return 'down'
    case 'down':
      return 'up'
    case 'left':
      return 'right'
    case 'right':
      return 'left'
  }
}

async function main () {
  const mtx = await readFile('./matrix.txt', handleMatrix)
  const cmd = await readFile('./command.txt', handleCommands)
  const cLen = cmd.length
  const mRow = mtx.length
  const mCol = mtx[0].length
  let row = 0
  let col = 0
  for (let steps of cmd) {
    let direction = steps[0]
    let step = steps[1]
    if (direction === 'left') {
      direction = 'right'
      step *= -1
    } else if (direction === 'up') {
      direction = 'down'
      step *= -1
    }

    if (direction === 'down') {
      let temp = (row + step) % mRow
      if (temp < 0) {
        row = temp + mRow
      } else {
        row = temp
      }
    } else if (direction === 'right') {
      let temp = (col + step) % mCol
      if (temp < 0) {
        col = temp + mCol
      } else {
        col = temp
      }
    }
  }
  console.log(row + 1, col + 1, matrix[row][col])
}

main()
